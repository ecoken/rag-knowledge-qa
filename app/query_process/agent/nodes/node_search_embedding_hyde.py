"""节点：切片搜索-假设性文档 (node_search_embedding_hyde)

先让模型对问题写一段「假想的标准答案」，再拿这段答案去检索，
而不是拿问题本身去检索。这就是 HyDE（Hypothetical Document Embeddings）。

**为什么这样反而更准**：向量检索本质是在比较两段文本的相似度，
但问题和答案在文本形态上差异很大。用户问「打印出来有横条纹怎么办」，
手册里对应的段落写的是「若打印结果出现规律性横向条纹，请执行打印头
清洗程序，并检查感光鼓是否磨损」。问句和陈述句、口语和术语，
直接比对相似度并不高。

而模型凭常识写出的假设性答案是陈述句、带专业术语的——形态上和手册段落
同构，向量距离自然更近。**即使模型编造的细节是错的也没关系**：我们只用
它来做检索，最终答案仍然完全基于真实检索到的手册内容生成。

代价是多一次 LLM 调用的延迟。所以它和常规检索并行执行而非串行，
两路耗时重叠，端到端只多出一次模型调用的时间。
"""

import sys

from langchain_core.messages import HumanMessage

from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.lm.llm_utils import get_llm_client
from app.query_process.agent.nodes._retrieval import retrieve
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task


def node_search_embedding_hyde(state: QueryGraphState) -> dict:
    """只返回本节点负责的字段，理由同 node_search_embedding：
    并行分支返回完整 state 会触发 LangGraph 的并发写入冲突。
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}] 开始 HyDE 检索")
    add_running_task(state["task_id"], function_name, is_stream=True)

    query_text = state.get("rewritten_query") or state.get("query") or ""
    try:
        prompt = load_prompt("hyde_prompt", rewritten_query=query_text)
        llm = get_llm_client()
        response = llm.invoke([HumanMessage(content=prompt)])
        hyde_doc = (response.content or "").strip()

        search_text = hyde_doc
        if not hyde_doc:
            # 模型没产出范文时退回原问题检索。此时本路与常规路结果高度重合，
            # RRF 会把重复项合并，不会造成结果污染。
            logger.warning(f"[{function_name}] HyDE 范文为空，退回原问题检索")
            search_text = query_text
        else:
            logger.info(f"[{function_name}] HyDE 范文（前 60 字）：{hyde_doc[:60]}...")

        results = retrieve(search_text, state.get("item_names") or [])
        logger.success(f"[{function_name}] HyDE 检索召回 {len(results)} 条")
        return {"hyde_doc": hyde_doc, "recall_hyde": results}

    except Exception as e:
        # HyDE 是增强路径，失败时静默降级：常规路仍然提供完整的召回结果
        logger.error(f"[{function_name}] HyDE 检索失败，本路返回空：{e}", exc_info=True)
        return {"hyde_doc": "", "recall_hyde": []}
    finally:
        add_done_task(state["task_id"], function_name, is_stream=True)
