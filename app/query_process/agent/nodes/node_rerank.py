"""节点：重排序 (node_rerank)

用交叉编码器（bge-reranker-large）对融合后的候选做精排，取前若干条作为
最终送进大模型的上下文。

**双塔 vs 交叉编码器，为什么两者都要**：
前面的向量检索是双塔模型——问题和切片各自独立编码成向量，再比余弦距离。
好处是切片向量可以预先算好存库，检索时只查一次，快；
代价是问题和切片从头到尾没有交互过，模型无法判断「这段话是否真的回答了
这个问题」，只能判断「它们大体上是不是一个话题」。

交叉编码器把问题和切片拼成一条输入一起过模型，逐层做注意力交互，
判别精度高得多。代价是无法预计算——每个候选都要单独推理一次，
对全库做这件事是不可能的。

所以标准做法是两段式：**双塔负责从海量数据里快速捞出几十条候选（保召回），
交叉编码器负责在这几十条里挑出真正对题的几条（保精度）**。
这是整条检索链路上性价比最高的一个环节。
"""

import sys

from app.core.logger import logger
from app.lm.reranker_utils import rerank_scores
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task

# 精排后送进大模型的切片数。太少容易漏掉答案所在片段，
# 太多则引入噪声、推高成本，还可能触发「中间迷失」——
# 模型对长上下文中部的内容注意力明显偏低。
TOP_K = 5
# 相关性分数下限。低于该阈值的切片即便排进前 K 也丢弃，
# 避免知识库里压根没有相关内容时，硬凑五条无关片段误导模型。
SCORE_THRESHOLD = 0.0


def node_rerank(state: QueryGraphState) -> QueryGraphState:
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}] 开始精排")
    add_running_task(state["task_id"], function_name, is_stream=True)

    candidates = state.get("fused_chunks") or []
    try:
        if not candidates:
            logger.warning(f"[{function_name}] 无候选切片，跳过精排")
            state["reranked_chunks"] = []
            return state

        query_text = state.get("rewritten_query") or state.get("query") or ""

        # 交叉编码器输入格式：[[查询, 文档], ...]，一次性批量打分。
        # 走 rerank_scores 而非直接调模型：推理需要串行化，
        # 多线程并发调用同一个 FP16 模型实例会抛 dtype 错误。
        pairs = [[query_text, c.get("content", "")] for c in candidates]
        scores = rerank_scores(pairs, normalize=True)

        scored = []
        for chunk, score in zip(candidates, scores):
            item = dict(chunk)
            item["rerank_score"] = float(score)
            scored.append(item)

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        top = [c for c in scored[:TOP_K] if c["rerank_score"] >= SCORE_THRESHOLD]
        state["reranked_chunks"] = top

        if top:
            logger.success(
                f"[{function_name}] 精排完成：{len(candidates)} 条候选 → 保留 {len(top)} 条，"
                f"最高分 {top[0]['rerank_score']:.4f}，最低分 {top[-1]['rerank_score']:.4f}"
            )
            for i, c in enumerate(top, 1):
                logger.info(
                    f"    [{i}] {c['rerank_score']:.4f} | {c.get('item_name', '')} "
                    f"| {c.get('parent_title', '')}/{c.get('title', '')}"
                )
        else:
            logger.warning(f"[{function_name}] 所有候选分数均低于阈值，判定知识库无相关内容")

    except Exception as e:
        # 精排失败时退化为直接取 RRF 融合的前 TOP_K 条。
        # 精度会降，但仍是经过两路召回和融合的结果，远好于返回空。
        logger.error(f"[{function_name}] 精排失败，退化使用 RRF 排序结果：{e}", exc_info=True)
        state["reranked_chunks"] = candidates[:TOP_K]
    finally:
        add_done_task(state["task_id"], function_name, is_stream=True)

    return state
