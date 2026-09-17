"""节点：导入向量库 (node_import_milvus)

把向量化好的切片写入 Milvus，并保证重复导入同一份文档时结果幂等。

三件事按顺序做：

1. **确保集合存在**。集合的 schema 在这里定义——稠密向量维度不写死，
   而是从实际生成的向量里读，避免 .env 里的 EMBEDDING_DIM 配错
   （BGE-M3 是 1024 维）导致建表和写入对不上。

2. **按 item_name 先删后插**。同一份手册改几个字重新导入，如果直接插入，
   Milvus 里会同时存在新旧两份切片。检索时同一段内容占掉多个名额，
   把本该召回的其他片段挤出去——这类问题在结果里表现为「答案看起来
   没错但就是不全」，很难排查。先删干净是最省事的做法。

3. **分批插入**。单次插入上千条会超过 gRPC 消息大小限制，
   而且失败时整批回滚，重试代价高。

主键 chunk_id 用雪花式的自增分配：查询集合当前最大 id 往后排。
不用 auto_id 是因为查询侧需要靠 chunk_id 做父子块回查
（fetch_chunks_by_chunk_ids），显式主键更好控制。
"""

import sys

from pymilvus import DataType

from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.utils.task_utils import add_done_task, add_running_task

# 单批插入条数。Milvus 默认 gRPC 消息上限 64MB，
# 一条记录含 1024 维浮点向量约 4KB，200 条留足安全余量。
INSERT_BATCH_SIZE = 200
# content 字段最大长度，超出部分截断。切分节点已控制在 800 字符左右，
# 这里设 8192 纯粹是兜底，防止异常数据把插入整批打挂。
MAX_CONTENT_LENGTH = 8192


def _ensure_collection(client, collection_name: str, dim: int) -> None:
    """集合不存在则按既定 schema 创建，并为两种向量分别建索引。"""
    if client.has_collection(collection_name):
        return

    logger.info(f"集合 [{collection_name}] 不存在，开始创建（稠密维度 {dim}）")
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("chunk_id", DataType.INT64, is_primary=True)
    schema.add_field("content", DataType.VARCHAR, max_length=MAX_CONTENT_LENGTH)
    schema.add_field("title", DataType.VARCHAR, max_length=512)
    schema.add_field("parent_title", DataType.VARCHAR, max_length=512)
    schema.add_field("item_name", DataType.VARCHAR, max_length=256)
    schema.add_field("image_urls", DataType.VARCHAR, max_length=4096)
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

    index_params = client.prepare_index_params()
    # 稠密向量用 COSINE：BGE-M3 已做 L2 归一化，余弦与内积等价，
    # 但显式写 COSINE 可读性更好，也与检索侧的 dense_params 保持一致
    index_params.add_index(
        field_name="dense_vector", index_type="AUTOINDEX", metric_type="COSINE"
    )
    # 稀疏向量只支持 IP（内积），这是 Milvus 的硬约束
    index_params.add_index(
        field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP"
    )

    client.create_collection(
        collection_name=collection_name, schema=schema, index_params=index_params
    )
    logger.success(f"集合 [{collection_name}] 创建完成")


def _next_chunk_id(client, collection_name: str) -> int:
    """取当前集合最大 chunk_id + 1 作为本次插入的起始主键。"""
    try:
        rows = client.query(
            collection_name=collection_name,
            filter="chunk_id >= 0",
            output_fields=["chunk_id"],
            limit=1,
            # 按主键倒序取第一条即为最大值
            sort_by_field="chunk_id",
            sort_order="desc",
        )
        if rows:
            return int(rows[0]["chunk_id"]) + 1
    except Exception as e:
        # 旧版本 Milvus 不支持 sort_by_field，退化成全量扫描取最大值。
        # 集合规模有限（单机手册库），这个代价可以接受。
        logger.warning(f"按主键倒序查询失败，回退全量扫描取最大 chunk_id：{e}")
        try:
            rows = client.query(
                collection_name=collection_name,
                filter="chunk_id >= 0",
                output_fields=["chunk_id"],
                limit=16384,
            )
            if rows:
                return max(int(r["chunk_id"]) for r in rows) + 1
        except Exception as inner:
            logger.warning(f"全量扫描同样失败，主键从 0 开始：{inner}")
    return 0


def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}] 开始执行，文件：{state.get('file_title')}")
    add_running_task(state["task_id"], function_name)

    try:
        records = state.get("embeddings_content") or []
        if not records:
            logger.error(f"[{function_name}] embeddings_content 为空，无数据可入库")
            return state

        client = get_milvus_client()
        if client is None:
            raise RuntimeError("Milvus 客户端不可用，请检查 MILVUS_URL 配置与服务状态")

        collection_name = milvus_config.chunks_collection
        if not collection_name:
            raise ValueError("缺少 CHUNKS_COLLECTION 环境变量配置")

        dim = len(records[0]["dense_vector"])
        _ensure_collection(client, collection_name, dim)

        # --- 幂等：先按 item_name 清掉这份文档的旧切片 ---
        item_name = records[0].get("item_name") or ""
        if item_name:
            expr = f'item_name == "{escape_milvus_string(item_name)}"'
            try:
                deleted = client.delete(collection_name=collection_name, filter=expr)
                logger.info(f"[{function_name}] 已清理 [{item_name}] 的旧切片：{deleted}")
            except Exception as e:
                # 首次导入时集合为空，删除失败属正常情况，不应阻断插入
                logger.warning(f"[{function_name}] 清理旧切片未生效（可能是首次导入）：{e}")

        # --- 分批插入 ---
        next_id = _next_chunk_id(client, collection_name)
        rows = []
        for offset, rec in enumerate(records):
            rows.append({
                "chunk_id": next_id + offset,
                "content": rec["content"][:MAX_CONTENT_LENGTH],
                "title": rec.get("title", "")[:512],
                "parent_title": rec.get("parent_title", "")[:512],
                "item_name": (rec.get("item_name") or "")[:256],
                # 图片 URL 列表序列化成换行分隔的字符串，查询侧再拆回来。
                # 不用 JSON 字段是为了兼容旧版本 Milvus 的标量类型支持。
                "image_urls": "\n".join(rec.get("image_urls") or [])[:4096],
                "dense_vector": rec["dense_vector"],
                "sparse_vector": rec["sparse_vector"],
            })

        inserted = 0
        total_batches = (len(rows) + INSERT_BATCH_SIZE - 1) // INSERT_BATCH_SIZE
        for bi, start in enumerate(range(0, len(rows), INSERT_BATCH_SIZE), 1):
            batch = rows[start:start + INSERT_BATCH_SIZE]
            client.insert(collection_name=collection_name, data=batch)
            inserted += len(batch)
            logger.info(f"[{function_name}] 入库进度 {bi}/{total_batches}（累计 {inserted} 条）")

        # flush 让数据立即可检索。不调用的话新数据要等后台自动落盘，
        # 导入完马上查会查不到，很容易被误判成入库失败。
        client.flush(collection_name=collection_name)
        logger.success(
            f"[{function_name}] 入库完成：[{item_name}] 共 {inserted} 条，"
            f"主键区间 [{next_id}, {next_id + inserted - 1}]"
        )

    except Exception as e:
        logger.error(f"[{function_name}] 导入 Milvus 失败：{e}", exc_info=True)
        raise
    finally:
        add_done_task(state["task_id"], function_name)

    return state


if __name__ == "__main__":
    from app.import_process.agent.state import create_default_state

    st = create_default_state(
        task_id="test_import_milvus",
        item_name="测试设备",
        embeddings_content=[{
            "content": "打印机卡纸时，请先断电再打开后盖取出纸张。",
            "title": "卡纸处理", "parent_title": "故障排除",
            "item_name": "测试设备", "image_urls": [],
            "dense_vector": [0.01] * 1024, "sparse_vector": {1: 0.5, 7: 0.8},
        }],
    )
    node_import_milvus(st)
