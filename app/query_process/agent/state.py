"""查询流程的状态定义。

与导入流程的 ImportGraphState 分开定义，而不是复用同一个大字典：
两条链路关心的数据几乎没有交集，硬塞进一个 TypedDict 会让每个节点都要
面对一堆和自己无关的字段，类型提示也失去意义。
"""

import copy
from typing import TypedDict

from app.core.logger import logger


class QueryGraphState(TypedDict):
    """查询链路全程共享的状态。

    字段按流水线顺序排列，读一遍就能看出数据是怎么一步步演进的。
    """

    task_id: str                # 任务唯一 ID，用于 SSE 进度推送与日志追踪

    # --- 输入 ---
    query: str                  # 用户原始提问
    history: list               # 历史对话，[{"role": "user"/"assistant", "content": str}]

    # --- 问题理解（node_item_name_confirm）---
    item_names: list            # 识别出的商品名，用于 Milvus 标量过滤
    rewritten_query: str        # 指代消解后的独立完整问题

    # --- 多路召回 ---
    hyde_doc: str               # HyDE 生成的假设性答案范文
    recall_normal: list         # 用原问题直接检索的结果
    recall_hyde: list           # 用假设性文档检索的结果

    # --- 融合与精排 ---
    fused_chunks: list          # RRF 融合去重后的候选集
    reranked_chunks: list       # 交叉编码器精排后的最终上下文

    # --- 输出 ---
    answer: str                 # 生成的答案
    image_urls: list            # 答案引用到的图片 URL


query_default_state: QueryGraphState = {
    "task_id": "",
    "query": "",
    "history": [],
    "item_names": [],
    "rewritten_query": "",
    "hyde_doc": "",
    "recall_normal": [],
    "recall_hyde": [],
    "fused_chunks": [],
    "reranked_chunks": [],
    "answer": "",
    "image_urls": [],
}


def create_query_state(**overrides) -> QueryGraphState:
    """创建查询状态，支持按需覆盖字段。

    每次都从默认值深拷贝，避免多个请求共享同一个可变列表——
    并发场景下这类隐蔽的状态串味极难排查。
    """
    state = copy.deepcopy(query_default_state)
    state.update(overrides)
    return state


if __name__ == "__main__":
    st = create_query_state(task_id="demo", query="打印机卡纸了怎么办")
    logger.info(st)
