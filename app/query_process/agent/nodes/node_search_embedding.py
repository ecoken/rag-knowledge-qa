"""节点：切片搜索 (node_search_embedding)

用改写后的问题直接做混合检索——多路召回中的「常规路」。

这一路的价值在于忠实于用户的原始表述。当用户的问法本身就贴近手册用语
（「更换墨盒的步骤」对上标题「更换墨盒」），直接检索的命中质量极高，
不需要任何改写或扩展。HyDE 那一路反而可能因为模型编造的范文跑偏。

两路并行、各取所长，再由 RRF 融合，是比单路检索更稳的做法。
"""

import sys

from app.core.logger import logger
from app.query_process.agent.nodes._retrieval import retrieve
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task


def node_search_embedding(state: QueryGraphState) -> dict:
    """只返回本节点负责的字段，不返回整个 state。

    这一点在并行分支里是硬性要求：LangGraph 会把各分支的返回值合并进状态，
    若两个并行节点都返回完整 state，等于对每个字段都发起了并发写入，
    没有配 reducer 的字段会直接抛 InvalidUpdateError。只返回自己写的键，
    合并时天然无冲突。
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}] 开始常规检索")
    add_running_task(state["task_id"], function_name, is_stream=True)

    try:
        query_text = state.get("rewritten_query") or state.get("query") or ""
        results = retrieve(query_text, state.get("item_names") or [])
        logger.success(f"[{function_name}] 常规检索召回 {len(results)} 条")
        return {"recall_normal": results}
    except Exception as e:
        # 单路失败不影响另一路，融合节点会自动只用可用的那一路
        logger.error(f"[{function_name}] 常规检索失败，本路返回空：{e}", exc_info=True)
        return {"recall_normal": []}
    finally:
        add_done_task(state["task_id"], function_name, is_stream=True)
