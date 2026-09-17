"""节点：生成答案 (node_answer_output)

把精排后的切片拼成上下文交给大模型作答，并收集答案引用到的图片。

**上下文里为什么要带标题和来源**：只给正文，模型无法交代信息出自哪份手册、
哪一章。设备手册场景下用户常常需要核对原文，答案里能指出
「见《XX 用户手册》故障排除 / 卡纸处理」比单纯给结论有用得多，
也让模型更不容易把几份手册的内容混着说。

**知识库没有答案时必须说没有**。这类系统最危险的失败不是答不出来，
而是用相关但不对题的内容编一个看起来合理的答案——用户照着操作可能损坏设备。
所以精排结果为空时直接返回固定话术，根本不调用模型。
"""

import sys

from langchain_core.messages import HumanMessage

from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.lm.llm_utils import get_llm_client
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    add_done_task,
    add_running_task,
    set_task_result,
    update_task_status,
)

# 知识库确实没有相关内容时的固定回复。宁可明确说没有，也不让模型硬答。
NO_CONTEXT_ANSWER = "抱歉，知识库中没有检索到与该问题相关的内容，无法作答。请换个说法，或确认该设备的手册是否已导入。"
# 带入模型的历史轮数，与问题理解节点保持一致
MAX_HISTORY_TURNS = 6


def _build_context(chunks: list) -> str:
    """把切片拼成带来源标注的上下文块。"""
    blocks = []
    for i, c in enumerate(chunks, 1):
        source = " / ".join(
            x for x in (c.get("item_name"), c.get("parent_title"), c.get("title")) if x
        )
        header = f"【片段{i}｜来源：{source}】" if source else f"【片段{i}】"
        body = c.get("content", "")
        images = c.get("image_urls") or []
        if images:
            # 把可用图片随片段一起给模型，它才可能按 prompt 要求在答案末尾附图
            body += "\n可用图片：\n" + "\n".join(images)
        blocks.append(f"{header}\n{body}")
    return "\n\n".join(blocks)


def _format_history(history: list) -> str:
    if not history:
        return "（无历史对话）"
    lines = []
    for turn in history[-MAX_HISTORY_TURNS:]:
        role = "用户" if turn.get("role") == "user" else "助手"
        lines.append(f"{role}：{turn.get('content', '')}")
    return "\n".join(lines)


def node_answer_output(state: QueryGraphState) -> QueryGraphState:
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}] 开始生成答案")
    add_running_task(state["task_id"], function_name, is_stream=True)

    try:
        chunks = state.get("reranked_chunks") or []
        if not chunks:
            # 没有检索到任何相关内容，直接给出明确答复，不调用模型。
            # 这一步是幻觉防线：没有依据就不生成。
            logger.warning(f"[{function_name}] 无可用上下文，返回兜底答复")
            state["answer"] = NO_CONTEXT_ANSWER
            state["image_urls"] = []
            return state

        prompt = load_prompt(
            "answer_out",
            context=_build_context(chunks),
            history=_format_history(state.get("history") or []),
            item_names="、".join(state.get("item_names") or []) or "（未指定）",
            question=state.get("rewritten_query") or state.get("query") or "",
        )

        llm = get_llm_client()
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = (response.content or "").strip()

        state["answer"] = answer or NO_CONTEXT_ANSWER
        # 汇总本次引用到的全部图片，去重并保持出现顺序，供前端单独渲染
        seen, images = set(), []
        for c in chunks:
            for url in c.get("image_urls") or []:
                if url not in seen:
                    seen.add(url)
                    images.append(url)
        state["image_urls"] = images

        set_task_result(state["task_id"], "answer", state["answer"])
        update_task_status(state["task_id"], TASK_STATUS_COMPLETED)
        logger.success(
            f"[{function_name}] 答案生成完成：{len(state['answer'])} 字符，"
            f"引用 {len(chunks)} 个片段、{len(images)} 张图片"
        )

    except Exception as e:
        logger.error(f"[{function_name}] 答案生成失败：{e}", exc_info=True)
        state["answer"] = NO_CONTEXT_ANSWER
        state["image_urls"] = []
        set_task_result(state["task_id"], "error", str(e))
    finally:
        add_done_task(state["task_id"], function_name, is_stream=True)

    return state
