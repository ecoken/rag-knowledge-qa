"""节点：主体识别 (node_item_name_recognition)

识别这份文档讲的到底是哪个商品，结果写入 state["item_name"]。

这个字段有两个用途，都很关键：

1. **入库幂等**。同一份手册重复导入时，按 item_name 先删后插，
   避免 Milvus 里堆积重复切片——重复切片会让检索结果被同一段内容刷屏，
   挤掉本该召回的其他片段。

2. **检索过滤**。用户问「MateBook B3 怎么进 BIOS」时，查询侧先确定
   item_name，再用它做 Milvus 标量过滤，把检索范围从 87 份手册缩到 1 份。
   没有这层过滤，跨设备的相似章节（几乎每份手册都有「安全须知」）
   会严重互相干扰。

识别材料取文件名 + 正文开头若干切片：文件名往往已经包含型号，
正文开头则是封面和产品概述，两者互补。全文喂给模型既费 token 又没必要。
"""

import sys

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.lm.llm_utils import get_llm_client
from app.utils.task_utils import add_done_task, add_running_task

# 送进模型的切片数量。前几片通常是封面、目录和产品概述，
# 型号信息基本都在这个范围内，再多只是增加成本。
CONTEXT_CHUNK_COUNT = 5
# 单个切片截断长度，防止个别超长切片把上下文撑爆
CONTEXT_CHUNK_CHARS = 500


def _build_context(state: ImportGraphState) -> str:
    """拼接用于识别的正文片段；没有切片时退化为直接截取 Markdown 开头。"""
    chunks = state.get("chunks") or []
    if chunks:
        return "\n---\n".join(
            c.get("content", "")[:CONTEXT_CHUNK_CHARS]
            for c in chunks[:CONTEXT_CHUNK_COUNT]
        )
    return (state.get("md_content") or "")[:CONTEXT_CHUNK_CHARS * CONTEXT_CHUNK_COUNT]


def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}] 开始执行，文件：{state.get('file_title')}")
    add_running_task(state["task_id"], function_name)

    file_title = state.get("file_title") or ""
    try:
        context = _build_context(state)
        if not context.strip():
            # 没有任何正文可供判断时，退化成用文件名当主体名。
            # 这比留空好：至少幂等删除和检索过滤还能按文件名工作。
            logger.warning(f"[{function_name}] 无正文内容，回退使用文件名作为主体名")
            state["item_name"] = file_title
            return state

        prompt = load_prompt("item_name_recognition", file_title=file_title, context=context)
        system_prompt = load_prompt("product_recognition_system")

        llm = get_llm_client()
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ])
        item_name = (response.content or "").strip().strip('"').strip("'")

        # 模型偶尔会输出「商品名称：XXX」这类带前缀的内容，去掉冒号前缀
        if "：" in item_name and len(item_name.split("：", 1)[0]) <= 8:
            item_name = item_name.split("：", 1)[1].strip()

        # 模型识别失败或返回明显异常（太长说明它在解释而不是在回答）时回退文件名
        if not item_name or len(item_name) > 60:
            logger.warning(
                f"[{function_name}] 识别结果不可用（{item_name[:40]!r}），回退使用文件名"
            )
            item_name = file_title

        state["item_name"] = item_name
        logger.success(f"[{function_name}] 主体识别完成：{item_name}")

    except Exception as e:
        # 主体识别失败不应中断整条导入链路——退化成文件名后，
        # 切分好的内容照样能入库，只是检索过滤的精度略降。
        logger.error(f"[{function_name}] 主体识别失败，回退使用文件名：{e}", exc_info=True)
        state["item_name"] = file_title
    finally:
        add_done_task(state["task_id"], function_name)

    return state


if __name__ == "__main__":
    from app.import_process.agent.state import create_default_state

    st = create_default_state(
        task_id="test_item_name",
        file_title="HUAWEI MateBook B3-410 用户手册",
        chunks=[{"content": "本手册适用于华为 MateBook B3-410 笔记本电脑，介绍基本操作与维护。"}],
    )
    node_item_name_recognition(st)
    logger.info(f"识别结果：{st['item_name']}")
