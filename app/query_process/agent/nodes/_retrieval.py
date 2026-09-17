"""检索公共逻辑。

常规检索和 HyDE 检索只有「用什么文本去查」这一点不同，其余步骤完全一致：
向量化 → 拼过滤条件 → 混合检索 → 归一化结果。把共同部分收在这里，
两个节点各自只负责准备查询文本，避免同一段检索逻辑写两遍后改一处漏一处。

下划线开头表示这是包内实现细节，不作为节点对外暴露。
"""

from app.clients.milvus_utils import (
    create_hybrid_search_requests,
    get_milvus_client,
    hybrid_search,
)
from app.conf.milvus_config import milvus_config
from app.core.logger import logger
from app.lm.embedding_utils import generate_embeddings
from app.utils.escape_milvus_string_utils import escape_milvus_string

# 单路召回条数。取得比最终送进模型的上下文多得多，是因为后面还有
# RRF 融合和交叉编码器精排两道筛子——召回阶段宁滥勿缺，
# 精排阶段再做减法。召回不到的内容，后面再强的精排也救不回来。
RECALL_LIMIT = 20

# 稠密与稀疏的融合权重。稠密略高是因为设备手册的提问多为自然语言描述
# （「怎么清洁滤网」），语义匹配的贡献更大；但稀疏权重不能太低，
# 型号和错误码这类字面匹配全靠它。
DENSE_WEIGHT = 0.6
SPARSE_WEIGHT = 0.4

OUTPUT_FIELDS = ["chunk_id", "content", "title", "parent_title", "item_name", "image_urls"]


def build_item_filter(item_names: list) -> str | None:
    """把商品名列表拼成 Milvus 标量过滤表达式。

    没识别出商品时返回 None，表示全库检索——这是刻意的降级策略：
    宁可召回范围大一些，也不要因为过滤条件拼错而一条都召不回。
    """
    if not item_names:
        return None
    quoted = ", ".join(f'"{escape_milvus_string(n)}"' for n in item_names)
    return f"item_name in [{quoted}]"


def retrieve(query_text: str, item_names: list, *, limit: int = RECALL_LIMIT) -> list[dict]:
    """对给定文本做一次稠密+稀疏混合检索，返回归一化后的切片列表。

    :param query_text: 用于检索的文本（原问题或 HyDE 生成的假设性答案）
    :param item_names: 商品名过滤条件，空列表表示不过滤
    :param limit: 返回条数
    :return: [{chunk_id, content, title, parent_title, item_name, image_urls, score}, ...]
             任一环节失败都返回空列表，由调用方决定降级策略
    """
    if not query_text.strip():
        logger.warning("检索文本为空，跳过本路召回")
        return []

    client = get_milvus_client()
    if client is None:
        logger.error("Milvus 客户端不可用，本路召回返回空")
        return []

    collection = milvus_config.chunks_collection
    if not collection:
        logger.error("缺少 CHUNKS_COLLECTION 配置，本路召回返回空")
        return []

    try:
        vectors = generate_embeddings([query_text])
        dense_vector = vectors["dense"][0]
        sparse_vector = vectors["sparse"][0]

        expr = build_item_filter(item_names)
        reqs = create_hybrid_search_requests(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            expr=expr,
            limit=limit,
        )
        raw = hybrid_search(
            client,
            collection,
            reqs,
            ranker_weights=(DENSE_WEIGHT, SPARSE_WEIGHT),
            # 归一化后再加权：稠密走 COSINE（值域 -1~1），稀疏走 IP（值域无上界），
            # 不归一化的话稀疏分数量级会直接淹没稠密分数，权重形同虚设。
            norm_score=True,
            limit=limit,
            output_fields=OUTPUT_FIELDS,
        )
        if not raw:
            return []

        results: list[dict] = []
        for hit in raw[0]:
            entity = hit.get("entity", hit) if isinstance(hit, dict) else {}
            image_urls = entity.get("image_urls") or ""
            results.append({
                "chunk_id": entity.get("chunk_id"),
                "content": entity.get("content", ""),
                "title": entity.get("title", ""),
                "parent_title": entity.get("parent_title", ""),
                "item_name": entity.get("item_name", ""),
                # 入库时用换行拼接，这里拆回列表
                "image_urls": [u for u in image_urls.split("\n") if u.strip()],
                "score": hit.get("distance", 0.0) if isinstance(hit, dict) else 0.0,
            })

        logger.info(f"混合检索命中 {len(results)} 条，过滤条件：{expr or '（全库）'}")
        return results

    except Exception as e:
        logger.error(f"混合检索执行失败：{e}", exc_info=True)
        return []
