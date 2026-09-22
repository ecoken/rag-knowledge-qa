"""检索公共逻辑。

常规检索和 HyDE 检索只有「用什么文本去查」这一点不同，其余步骤完全一致：
向量化 → 拼过滤条件 → 混合检索 → 归一化结果。把共同部分收在这里，
两个节点各自只负责准备查询文本，避免同一段检索逻辑写两遍后改一处漏一处。

下划线开头表示这是包内实现细节，不作为节点对外暴露。
"""

import re

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

# 检索返回的标量字段。
# image_urls 不在其中：既有集合（由早期导入流程建立，已存 1143 个切片）
# 没有这个字段，而向 Milvus 请求任何一个不存在的字段都会让整次检索直接报错。
# 图片链接改由切片正文中的 Markdown 语法解析得到，见 extract_image_urls。
# file_title 是原始文件名，item_name 是识别出的设备名，两者都用于标注答案出处。
OUTPUT_FIELDS = ["chunk_id", "content", "title", "parent_title", "item_name", "file_title"]

# Markdown 图片语法，用于从正文里回收图片链接
_IMAGE_IN_TEXT = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def extract_image_urls(content: str) -> list[str]:
    """从切片正文中解析 Markdown 图片链接。

    导入阶段图片链接是内嵌在正文里的，集合没有单独的 image_urls 字段。
    与其为此重灌已有的上千条切片，不如在检索侧把链接解析出来——
    schema 迁移的成本远高于一次正则匹配。
    """
    return _IMAGE_IN_TEXT.findall(content or "")


# 商品名归一化的召回条数与相似度下限。
#
# 只取 Top1：实测第 2 名普遍落在 0.60~0.77，全是同品牌的其他型号
# （问 CB304n 网桥会带出 CC 系列集中器），把它们纳入过滤只会稀释结果。
#
# 阈值 0.70 来自实测分布，两侧各留约 0.10 余量：
#   库中真实设备 Top1  0.8021 ~ 0.9955（最低是 NR-1200W）
#   库中不存在的设备    0.4459 ~ 0.5990（最高是「小米路由器」）
# 阈值定低了，问「小米路由器」会被归一化到 H3C 设备上，
# 系统于是拿 H3C 的配置步骤去回答一台库里根本没有的设备——
# 这类张冠李戴比直接答不出来危险得多。
ITEM_RESOLVE_TOPK = 1
ITEM_RESOLVE_MIN_SCORE = 0.70


def resolve_item_names(raw_names: list) -> list[str]:
    """把 LLM 识别出的自然语言商品名映射为库中的规范名。

    这一步是必需的，不是锦上添花：模型识别出的是「Aolynk CB304n 网桥」
    这种带空格的口语写法，而入库时经主体识别节点规范化后存的是
    「AolynkCB304nCable网桥」。Milvus 的标量过滤是精确字符串匹配，
    两者对不上就会过滤掉全部结果——症状是检索返回 0 条、
    系统直接走兜底拒答，而日志里看不到任何报错，极难排查。

    kb_item_names 集合正是为此存在：它为每份文档存了一条带向量的
    商品名记录，用向量检索即可完成口语名到规范名的映射。
    """
    if not raw_names:
        return []

    client = get_milvus_client()
    collection = milvus_config.item_name_collection
    if client is None or not collection:
        logger.warning("商品名归一化不可用，退回使用原始名称")
        return list(raw_names)

    resolved: list[str] = []
    try:
        vectors = generate_embeddings(list(raw_names))
        for raw, dense in zip(raw_names, vectors["dense"]):
            hits = client.search(
                collection_name=collection,
                data=[dense],
                anns_field="dense_vector",
                search_params={"metric_type": "COSINE"},
                limit=ITEM_RESOLVE_TOPK,
                output_fields=["item_name", "file_title"],
            )
            matched = [
                h["entity"]["item_name"] for h in (hits[0] if hits else [])
                if h.get("distance", 0) >= ITEM_RESOLVE_MIN_SCORE
            ]
            if matched:
                logger.info(f"商品名归一化：{raw!r} → {matched}")
                resolved.extend(matched)
            else:
                logger.info(f"商品名 {raw!r} 未匹配到库中任何设备，本次不按它过滤")
    except Exception as e:
        logger.warning(f"商品名归一化失败，退回使用原始名称：{e}")
        return list(raw_names)

    return list(dict.fromkeys(resolved))


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

        def _search(expr: str | None):
            reqs = create_hybrid_search_requests(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                expr=expr,
                limit=limit,
            )
            return hybrid_search(
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

        expr = build_item_filter(item_names)
        raw = _search(expr)

        # 带过滤却一条都没召回时，去掉过滤重试一次。
        # 过滤条件写错（商品名对不上库中取值）的表现就是静默返回空集，
        # 日志里没有任何报错，最终用户只会看到「知识库中没有相关内容」——
        # 明明有答案却答不出来，比范围大一些糟糕得多。
        if expr and not (raw and raw[0]):
            logger.warning(f"按 {expr} 过滤后召回为空，降级为全库检索")
            raw = _search(None)

        if not raw:
            return []

        results: list[dict] = []
        for hit in raw[0]:
            entity = hit.get("entity", hit) if isinstance(hit, dict) else {}
            content = entity.get("content", "")
            results.append({
                "chunk_id": entity.get("chunk_id"),
                "content": content,
                "title": entity.get("title", ""),
                "parent_title": entity.get("parent_title", ""),
                "item_name": entity.get("item_name", ""),
                "file_title": entity.get("file_title", ""),
                "image_urls": extract_image_urls(content),
                "score": hit.get("distance", 0.0) if isinstance(hit, dict) else 0.0,
            })

        logger.info(f"混合检索命中 {len(results)} 条，过滤条件：{expr or '（全库）'}")
        return results

    except Exception as e:
        logger.error(f"混合检索执行失败：{e}", exc_info=True)
        return []
