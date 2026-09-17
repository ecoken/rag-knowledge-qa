"""节点：文档切分 (node_document_split)

把 Markdown 全文切成适合向量化与检索的片段。

为什么按标题层级切而不是按固定长度切：
设备手册的语义边界天然由标题划定——「3.2 接线说明」下的内容属于同一主题，
从中间硬切会让一个片段里出现半句话，检索命中后送给模型的上下文是残缺的。
按标题切还能顺带拿到 title / parent_title 两级标题，既可以拼进向量化文本
提升召回（用户问「怎么换墨盒」，标题里的「更换墨盒」是极强的信号），
也可以在答案里向用户交代内容出处。

超长章节仍需二次切分：有些手册一节能有上万字，既超出 embedding 模型的
输入上限，也会把向量语义稀释成一团浆糊。这里按字符数滑窗切分并保留重叠，
避免把跨越切点的句子彻底割裂。
"""

import json
import re
import sys
from pathlib import Path

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_done_task, add_running_task

# 单个切片的目标字符数。设备手册以中文为主，一个汉字约占 1 token，
# 800 字符在 BGE-M3 的 8192 上限内留足余量，也便于重排序模型逐条打分。
MAX_CHUNK_CHARS = 800
# 相邻切片的重叠字符数，保证跨切点的句子在两个片段里都能读到完整语义
CHUNK_OVERLAP_CHARS = 120
# 小于该长度的片段并入上一片，避免产生「## 附录」这类只有标题没有正文的噪声切片
MIN_CHUNK_CHARS = 60

# Markdown ATX 标题：# 到 ###### 六级
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
# Markdown 图片：![alt](url)
_IMAGE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
# 断句优先级：从段落边界到句末标点，越靠前越优先
_SEPARATORS = ("\n\n", "\n", "。", "；", "！", "？", ". ")


def _split_long_text(text: str) -> list[str]:
    """把超长文本按字符数滑窗切分，相邻片段保留重叠。

    优先在段落或句子边界断开，找不到合适边界时才硬切，
    避免为了凑长度把一句话从中间劈开。
    """
    if len(text) <= MAX_CHUNK_CHARS:
        return [text] if text.strip() else []

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = start + MAX_CHUNK_CHARS
        if end >= len(text):
            pieces.append(text[start:])
            break

        # 在目标切点前回溯找边界，最多回退 1/3 个切片长度，
        # 回退太多会让切片长度参差不齐，反而影响检索稳定性
        floor = end - MAX_CHUNK_CHARS // 3
        cut = -1
        for sep in _SEPARATORS:
            found = text.rfind(sep, floor, end)
            if found > cut:
                cut = found + len(sep)
        if cut <= start:
            cut = end  # 整段没有任何边界，只能硬切

        pieces.append(text[start:cut])

        # 下一片回退一个重叠量；若回退后不前进则放弃重叠，防止死循环
        next_start = cut - CHUNK_OVERLAP_CHARS
        start = cut if next_start <= start else next_start

    return [p.strip() for p in pieces if p.strip()]


def _iter_sections(md_content: str):
    """遍历 Markdown，产出 (标题, 父标题, 正文) 三元组。

    用一个标题栈维护层级：遇到 N 级标题时，把栈里所有层级 >= N 的标题弹出，
    栈顶剩下的就是它的父标题。这样即使文档层级跳跃（从 # 直接跳到 ###），
    父子关系也不会错乱——手册类文档的标题层级往往并不规范。
    """
    stack: list[tuple[int, str]] = []  # [(层级, 标题)]
    current_title = ""
    current_parent = ""
    buffer: list[str] = []

    for line in md_content.splitlines():
        m = _HEADING.match(line)
        if not m:
            buffer.append(line)
            continue

        # 遇到新标题，先把上一节攒下的内容产出
        body = "\n".join(buffer).strip()
        if body:
            yield current_title, current_parent, body
        buffer = []

        level, title = len(m.group(1)), m.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        current_parent = stack[-1][1] if stack else ""
        current_title = title
        stack.append((level, title))

    body = "\n".join(buffer).strip()
    if body:
        yield current_title, current_parent, body


def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """把 md_content 切成 chunks，写入 state['chunks'] 并落盘一份便于排查。"""
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}] 开始执行，文件：{state.get('file_title')}")
    add_running_task(state["task_id"], function_name)

    try:
        md_content = state.get("md_content") or ""
        if not md_content.strip():
            logger.error(f"[{function_name}] md_content 为空，无法切分")
            return state

        chunks: list[dict] = []
        for title, parent_title, body in _iter_sections(md_content):
            # 图片链接单独抽出来随切片存储：答案生成时可按需引用配图，
            # 同时把图片语法从正文剔除，避免长串 URL 噪声污染向量
            image_urls = _IMAGE.findall(body)
            text_body = _IMAGE.sub("", body).strip()
            if not text_body:
                continue

            for piece in _split_long_text(text_body):
                # 标题拼进正文一起向量化：用户提问的措辞往往更接近标题而非正文
                heading_prefix = " / ".join(x for x in (parent_title, title) if x)
                content = f"{heading_prefix}\n{piece}" if heading_prefix else piece
                chunks.append({
                    "content": content,
                    "title": title,
                    "parent_title": parent_title,
                    "image_urls": image_urls,
                })

        # 合并过短切片，减少只有标题没有实质内容的噪声片段
        merged: list[dict] = []
        for c in chunks:
            if merged and len(c["content"]) < MIN_CHUNK_CHARS:
                merged[-1]["content"] += "\n" + c["content"]
                merged[-1]["image_urls"] = merged[-1]["image_urls"] + c["image_urls"]
            else:
                merged.append(c)

        state["chunks"] = merged
        avg = sum(len(c["content"]) for c in merged) // max(len(merged), 1)
        logger.success(f"[{function_name}] 切分完成：{len(merged)} 个切片，平均 {avg} 字符")

        # 落盘一份切片明细。调优切分策略时，这份文件是判断切得好不好的主要依据，
        # 比翻日志直观得多。
        local_dir = state.get("local_dir")
        if local_dir:
            split_path = Path(local_dir) / f"{state.get('file_title') or 'document'}_chunks.json"
            split_path.parent.mkdir(parents=True, exist_ok=True)
            split_path.write_text(
                json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            state["split_path"] = str(split_path)
            logger.info(f"[{function_name}] 切片明细已写入 {split_path}")

    except Exception as e:
        logger.error(f"[{function_name}] 文档切分失败：{e}", exc_info=True)
        raise
    finally:
        add_done_task(state["task_id"], function_name)

    return state


if __name__ == "__main__":
    from app.import_process.agent.state import create_default_state

    demo_md = """# 用户手册
## 1. 产品概述
本设备为多功能一体机，支持打印、复印、扫描三种功能。
## 2. 安装说明
### 2.1 拆箱
请确认包装内含主机、电源线、说明书各一份，如有缺失请联系售后。
![拆箱示意图](images/unbox.png)
### 2.2 接线
先连接电源线，确认指示灯亮起后，再连接 USB 数据线到电脑。
"""
    st = create_default_state(task_id="test_split", md_content=demo_md, file_title="demo")
    node_document_split(st)
    for i, c in enumerate(st["chunks"], 1):
        logger.info(f"[{i}] 父标题={c['parent_title']!r} 标题={c['title']!r} 图片={c['image_urls']}")
        logger.info(f"    {c['content'][:70]}...")
