"""查询流程主图。

    START
      └→ node_item_name_confirm        确认问题产品 + 指代消解改写
           ├→ node_search_embedding         常规混合检索  ┐
           └→ node_search_embedding_hyde    HyDE 混合检索  ┤ 并行
                └→ node_rrf                 倒排融合去重   ┘
                     └→ node_rerank         交叉编码器精排
                          └→ node_answer_output  生成答案
                               └→ END

两路检索并行而非串行：HyDE 需要额外一次 LLM 调用生成假设性文档，
串行执行会把这段延迟直接叠加到端到端耗时上。并行后两路耗时重叠，
总延迟约等于较慢的那一路。

节点命名与 app/utils/task_utils.py 里的中文映射表一一对应，
SSE 进度推送因此可以直接复用，前端无需额外配置。
"""

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from app.core.logger import logger
from app.query_process.agent.nodes.node_answer_output import node_answer_output
from app.query_process.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.query_process.agent.nodes.node_rerank import node_rerank
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
from app.query_process.agent.nodes.node_search_embedding_hyde import (
    node_search_embedding_hyde,
)
from app.query_process.agent.state import QueryGraphState

load_dotenv()

workflow = StateGraph(QueryGraphState)

workflow.add_node("node_item_name_confirm", node_item_name_confirm)
workflow.add_node("node_search_embedding", node_search_embedding)
workflow.add_node("node_search_embedding_hyde", node_search_embedding_hyde)
workflow.add_node("node_rrf", node_rrf)
workflow.add_node("node_rerank", node_rerank)
workflow.add_node("node_answer_output", node_answer_output)

workflow.add_edge(START, "node_item_name_confirm")

# 扇出：问题理解完成后，两路检索并行启动
workflow.add_edge("node_item_name_confirm", "node_search_embedding")
workflow.add_edge("node_item_name_confirm", "node_search_embedding_hyde")

# 扇入：两路都跑完后才进入融合。LangGraph 会自动等待所有入边完成，
# 不需要手动写同步逻辑。
workflow.add_edge("node_search_embedding", "node_rrf")
workflow.add_edge("node_search_embedding_hyde", "node_rrf")

workflow.add_edge("node_rrf", "node_rerank")
workflow.add_edge("node_rerank", "node_answer_output")
workflow.add_edge("node_answer_output", END)

kb_query_app = workflow.compile()


def answer_question(query: str, history: list | None = None, task_id: str = "") -> dict:
    """对外的同步问答入口。

    :param query: 用户提问
    :param history: 历史对话 [{"role": "user"/"assistant", "content": str}]
    :param task_id: 任务 ID，用于 SSE 进度推送；不传则按问题内容生成一个
    :return: {"answer": str, "image_urls": list, "chunks": list}
    """
    from app.query_process.agent.state import create_query_state

    state = create_query_state(
        task_id=task_id or f"query_{abs(hash(query)) % (10 ** 10)}",
        query=query,
        history=history or [],
    )
    result = kb_query_app.invoke(state)
    return {
        "answer": result.get("answer", ""),
        "image_urls": result.get("image_urls", []),
        # 把最终引用的切片一并返回，便于前端展示出处、也便于评测脚本算召回率
        "chunks": result.get("reranked_chunks", []),
    }


if __name__ == "__main__":
    logger.info("===== 查询流程联调 =====")
    out = answer_question("打印机卡纸了怎么处理")
    logger.info(f"答案：{out['answer']}")
    logger.info(f"引用图片：{out['image_urls']}")
    for c in out["chunks"]:
        logger.info(f"  [{c.get('rerank_score', 0):.4f}] {c.get('item_name')} / {c.get('title')}")
