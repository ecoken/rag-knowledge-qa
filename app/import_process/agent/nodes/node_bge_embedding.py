"""节点：向量化 (node_bge_embedding)

用 BGE-M3 把每个切片同时转成稠密向量和稀疏向量，组装成可直接写入
Milvus 的记录，存入 state["embeddings_content"]。

为什么要同时产出两种向量：
- **稠密向量**擅长语义匹配。用户问「机器卡纸了怎么办」，能召回标题写着
  「清除卡住的介质」的段落——字面零重合，语义高度相关。
- **稀疏向量**擅长字面匹配，本质是可学习的 BM25。设备手册里充斥着
  `KFR-35GW`、`E-05` 这类型号和错误码，稠密向量对这种无语义的符号串
  区分度很差，而稀疏向量能精确命中。

BGE-M3 的独特之处正是一次前向传播同时产出这两种向量，省掉了维护两套
模型和两次编码的开销。检索时再用 WeightedRanker 把两路结果加权融合，
兼顾语义和字面。

分批编码而不是一次性全量：整本手册可能切出上千片，一次性喂给模型
会把显存打爆，尤其在开了 FP16 的 GPU 上。
"""

import sys

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import generate_embeddings
from app.utils.task_utils import add_done_task, add_running_task

# 每批送入模型的切片数。单批过大易触发显存溢出，过小则浪费并行能力，
# 32 是在消费级显卡上比较稳妥的折中值。
EMBEDDING_BATCH_SIZE = 32


def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}] 开始执行，文件：{state.get('file_title')}")
    add_running_task(state["task_id"], function_name)

    try:
        chunks = state.get("chunks") or []
        if not chunks:
            logger.error(f"[{function_name}] chunks 为空，无内容可向量化")
            return state

        item_name = state.get("item_name") or state.get("file_title") or ""
        texts = [c.get("content", "") for c in chunks]

        dense_all: list[list[float]] = []
        sparse_all: list[dict] = []
        total_batches = (len(texts) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

        for bi, start in enumerate(range(0, len(texts), EMBEDDING_BATCH_SIZE), 1):
            batch = texts[start:start + EMBEDDING_BATCH_SIZE]
            logger.info(f"[{function_name}] 向量化进度 {bi}/{total_batches}（{len(batch)} 条）")
            vectors = generate_embeddings(batch)
            dense_all.extend(vectors["dense"])
            sparse_all.extend(vectors["sparse"])

        if len(dense_all) != len(chunks):
            raise ValueError(
                f"向量数量与切片数量不一致：{len(dense_all)} vs {len(chunks)}，疑似批处理丢数据"
            )

        # 组装 Milvus 记录。chunk_id 不在这里生成——主键要在入库节点
        # 按集合现状统一分配，避免多份文档并行导入时主键撞车。
        embeddings_content = []
        for chunk, dense, sparse in zip(chunks, dense_all, sparse_all):
            embeddings_content.append({
                "content": chunk.get("content", ""),
                "title": chunk.get("title", ""),
                "parent_title": chunk.get("parent_title", ""),
                "item_name": item_name,
                "image_urls": chunk.get("image_urls", []),
                "dense_vector": dense,
                "sparse_vector": sparse,
            })

        state["embeddings_content"] = embeddings_content
        dim = len(dense_all[0]) if dense_all else 0
        logger.success(
            f"[{function_name}] 向量化完成：{len(embeddings_content)} 条，稠密维度 {dim}，"
            f"稀疏平均非零维度 {sum(len(s) for s in sparse_all) // max(len(sparse_all), 1)}"
        )

    except Exception as e:
        logger.error(f"[{function_name}] 向量化失败：{e}", exc_info=True)
        raise
    finally:
        add_done_task(state["task_id"], function_name)

    return state


if __name__ == "__main__":
    from app.import_process.agent.state import create_default_state

    st = create_default_state(
        task_id="test_embedding",
        item_name="测试设备",
        chunks=[
            {"content": "打印机卡纸时，请先断电再打开后盖取出纸张。",
             "title": "卡纸处理", "parent_title": "故障排除", "image_urls": []},
            {"content": "错误码 E-05 表示墨盒未正确安装。",
             "title": "错误码", "parent_title": "故障排除", "image_urls": []},
        ],
    )
    node_bge_embedding(st)
    for rec in st["embeddings_content"]:
        logger.info(
            f"{rec['title']}: 稠密 {len(rec['dense_vector'])} 维，"
            f"稀疏 {len(rec['sparse_vector'])} 个非零维度"
        )
