"""节点：确认问题产品 (node_item_name_confirm)

做两件事：从对话中确定用户在问哪个商品，以及把带指代的问题改写成独立完整的问题。

**为什么必须先确定商品**：知识库里有 87 份设备手册，而几乎每份都有
「安全须知」「故障排除」「保修条款」这些同名章节。用户问「怎么清洁滤网」，
不做商品过滤的话，十几台不同型号空调的清洁章节会一起被召回，
精排也救不回来——它们在语义上确实都高度相关，只是答的不是用户那台机器。
拿到 item_name 后用 Milvus 标量过滤把范围锁到一份手册，
这是本项目里对准确率影响最大的一个环节。

**为什么要改写问题**：多轮对话里用户会说「那它的功耗呢」。
这句话单独拿去检索几乎召不回任何东西——没有主语、没有商品名。
结合历史改写成「MateBook B3-410 的功耗是多少」之后，
无论稠密还是稀疏检索都有了可匹配的实体。
"""

import json
import sys

from langchain_core.messages import HumanMessage

from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.lm.llm_utils import get_llm_client
from app.query_process.agent.nodes._retrieval import resolve_item_names
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task

# 带入模型的历史轮数。太多会稀释当前问题的权重，也徒增 token；
# 指代消解通常只依赖最近几轮。
MAX_HISTORY_TURNS = 6


def _format_history(history: list) -> str:
    """把历史对话拍平成文本。空历史返回明确提示，避免模板里出现空白。"""
    if not history:
        return "（无历史对话）"
    recent = history[-MAX_HISTORY_TURNS:]
    lines = []
    for turn in recent:
        role = "用户" if turn.get("role") == "user" else "助手"
        lines.append(f"{role}：{turn.get('content', '')}")
    return "\n".join(lines)


def _parse_response(raw: str) -> tuple[list, str]:
    """解析模型返回的 JSON。

    即便开了 json_mode，模型偶尔仍会用 ```json 包裹或在前后加解释，
    所以这里做一次容错剥离，而不是直接 json.loads 然后炸掉。
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text.lstrip("`")
        text = text.removeprefix("json").strip()
    # 退一步，从第一个 { 到最后一个 } 之间截取
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]

    data = json.loads(text)
    names = data.get("item_names") or []
    if isinstance(names, str):
        names = [names]
    names = [str(n).strip() for n in names if str(n).strip()]
    return names, str(data.get("rewritten_query") or "").strip()


def node_item_name_confirm(state: QueryGraphState) -> QueryGraphState:
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}] 开始执行，问题：{state.get('query')}")
    add_running_task(state["task_id"], function_name, is_stream=True)

    query = state.get("query") or ""
    try:
        prompt = load_prompt(
            "rewritten_query_and_itemnames",
            history_text=_format_history(state.get("history") or []),
            query=query,
        )
        # 开 json_mode 让模型直接吐结构化结果，比让它自由发挥再正则抠要稳得多
        llm = get_llm_client(json_mode=True)
        response = llm.invoke([HumanMessage(content=prompt)])
        item_names, rewritten = _parse_response(response.content)

        # 把模型识别出的口语商品名映射成库中的规范名。
        # 不做这一步，标量过滤会因为字符串对不上而过滤掉全部结果。
        canonical = resolve_item_names(item_names)

        # 改写失败时退回原问题：宁可少一层增强，也不能把空串送进检索
        state["item_names"] = canonical
        state["rewritten_query"] = rewritten or query

        logger.success(
            f"[{function_name}] 商品识别：{item_names or '（未识别）'} "
            f"→ 规范名 {canonical or '（未匹配，将全库检索）'}；"
            f"改写后问题：{state['rewritten_query']}"
        )

    except Exception as e:
        # 这一步失败不致命：不做商品过滤、直接用原问题检索，
        # 结果精度会下降但链路仍然走得通。
        logger.error(f"[{function_name}] 问题理解失败，降级为原问题全库检索：{e}", exc_info=True)
        state["item_names"] = []
        state["rewritten_query"] = query
    finally:
        add_done_task(state["task_id"], function_name, is_stream=True)

    return state


if __name__ == "__main__":
    from app.query_process.agent.state import create_query_state

    st = create_query_state(
        task_id="test_confirm",
        query="那它的功耗是多少",
        history=[
            {"role": "user", "content": "MateBook B3-410 怎么进 BIOS"},
            {"role": "assistant", "content": "开机时按 F2 即可进入 BIOS 设置界面。"},
        ],
    )
    node_item_name_confirm(st)
    logger.info(f"item_names={st['item_names']}  rewritten={st['rewritten_query']}")
