import sys
from pathlib import Path

import os

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task


def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 入口节点 (node_entry)
    为什么叫这个名字: 作为图的 Entry Point，负责接收外部输入并决定流程走向。
    未来要实现:
    1. 接收文件路径。
    2. 判断文件类型 (PDF/MD)。
    3. 设置 state 中的路由标记 (is_pdf_read_enabled / is_md_read_enabled)。
    """

    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}]开始执行了！现在的状态为:{state}")
    add_running_task(state["task_id"], function_name)

    local_file_path = state["local_file_path"]
    if not local_file_path:
        logger.error(f"[{function_name}]检查发现没有输入文件,无法继续解析!！")
        return state

    if local_file_path.endswith(".md"):
        state["is_md_read_enabled"] = True
        state["md_path"]=local_file_path

    elif local_file_path.endswith(".pdf"):
        state["is_pdf_read_enabled"] = True
        state["pdf_path"]=local_file_path
    else:
        logger.error(f"[{function_name}]文件格式不对,无法继续解析!！")
    file_title_os = os.path.basename(local_file_path).split(".")[0]
    file_title=Path(local_file_path).stem
    state["file_title"] = file_title
    logger.info(f">>> [{function_name}]开始结束了了！现在的状态为:{state}")
    add_done_task(state["task_id"], function_name)
    return state
