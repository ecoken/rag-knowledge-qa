"""节点：倒排融合 (node_rrf)

把常规检索和 HyDE 检索两路结果用 RRF（Reciprocal Rank Fusion）合并去重。

**为什么用 RRF 而不是直接比分数**：两路召回的相似度分数不可比。
常规路查的是问题、HyDE 路查的是一段范文，文本长度和语义密度都不同，
分数量级天然有差异，强行加权求和等于在比较两把刻度不同的尺子。

RRF 只看**名次**不看分数：某条切片在某一路里排第几，就贡献 1/(k+名次) 的分。
名次是跨路可比的——「在这一路里排第 1」和「在那一路里排第 1」含义一致。
这让 RRF 对分数分布免疫，也是它在多路召回融合中成为默认选择的原因。

常数 k=60 是原论文的经验值。它的作用是压平头部差距：
没有 k 时第 1 名得 1.0、第 2 名得 0.5，相差一倍，单路的头名会过度主导；
k=60 时两者是 1/61 和 1/62，差距很小，融合更看重「在多路中都靠前」
而不是「在某一路里排第一」——后者恰恰是我们想要的信号。
"""

import sys

from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task

# RRF 平滑常数，来自原论文 (Cormack et al., 2009) 的经验取值
RRF_K = 60
# 融合后保留的候选数，作为精排的输入。给精排留足选择空间，
# 但也不能太多——交叉编码器要对每条候选单独前向推理，条数直接决定耗时。
FUSION_LIMIT = 15


def _rrf_fuse(rankings: list[list[dict]], k: int = RRF_K) -> list[dict]:
    """对多路排序结果做 RRF 融合。

    :param rankings: 多路召回结果，每一路内部已按相关性降序排列
    :return: 按融合分降序排列的去重结果，每条附带 rrf_score 与 recall_paths
    """
    scores: dict = {}
    merged: dict = {}

    for path_index, ranking in enumerate(rankings):
        for rank, item in enumerate(ranking, start=1):
            # 用 chunk_id 作为去重键；缺失时退化用内容哈希，
            # 保证同一段内容不会因为主键缺失而在结果里出现两次
            key = item.get("chunk_id")
            if key is None:
                key = hash(item.get("content", ""))

            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in merged:
                merged[key] = dict(item)
                merged[key]["recall_paths"] = []
            merged[key]["recall_paths"].append(path_index)

    for key, item in merged.items():
        item["rrf_score"] = scores[key]

    return sorted(merged.values(), key=lambda x: x["rrf_score"], reverse=True)


def node_rrf(state: QueryGraphState) -> QueryGraphState:
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}] 开始融合多路召回")
    add_running_task(state["task_id"], function_name, is_stream=True)

    try:
        normal = state.get("recall_normal") or []
        hyde = state.get("recall_hyde") or []

        if not normal and not hyde:
            logger.warning(f"[{function_name}] 两路召回均为空，无内容可融合")
            state["fused_chunks"] = []
            return state

        fused = _rrf_fuse([normal, hyde])[:FUSION_LIMIT]
        state["fused_chunks"] = fused

        both = sum(1 for c in fused if len(set(c.get("recall_paths", []))) > 1)
        logger.success(
            f"[{function_name}] 融合完成：常规 {len(normal)} 条 + HyDE {len(hyde)} 条 "
            f"→ 去重后取前 {len(fused)} 条，其中 {both} 条被两路同时召回"
        )

    except Exception as e:
        # 融合失败时退化为直接拼接两路结果，保证链路不断
        logger.error(f"[{function_name}] RRF 融合失败，退化为直接拼接：{e}", exc_info=True)
        state["fused_chunks"] = ((state.get("recall_normal") or [])
                                 + (state.get("recall_hyde") or []))[:FUSION_LIMIT]
    finally:
        add_done_task(state["task_id"], function_name, is_stream=True)

    return state


if __name__ == "__main__":
    a = [{"chunk_id": 1, "content": "A"}, {"chunk_id": 2, "content": "B"},
         {"chunk_id": 3, "content": "C"}]
    b = [{"chunk_id": 3, "content": "C"}, {"chunk_id": 1, "content": "A"},
         {"chunk_id": 4, "content": "D"}]
    for item in _rrf_fuse([a, b]):
        logger.info(
            f"chunk_id={item['chunk_id']} rrf={item['rrf_score']:.5f} "
            f"命中路数={len(set(item['recall_paths']))}"
        )
