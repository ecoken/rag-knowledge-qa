"""节点：图片处理 (node_md_img)

把 MinerU 解析出来的本地图片上传到 MinIO，用多模态模型为每张图生成
中文描述，然后把 Markdown 里的本地相对路径替换成可公网访问的 URL。

为什么这一步对检索质量很重要：
设备手册里大量关键信息只存在于图片中——接线示意图、面板按键位置、
拆装步骤图。这些内容对纯文本检索完全不可见。给图片生成一句描述后，
描述会随所在段落一起被切分、向量化，用户问「电源接口在哪个位置」
就有机会命中那张面板图所在的片段，答案里再把图片 URL 带出来。

图片描述用上下文而不是只看图：单看一张接线图，模型只能说「这是一张
线路连接示意图」，信息量很低。把图片在原文中的上下各一段一起给它，
才能产出「HDMI 接口位于机身背面右侧」这种有检索价值的描述。

上传失败或模型调用失败都不中断流程：图片是增强项，为它牺牲整条
导入链路不划算。失败时保留原始链接，正文照常入库。
"""

import re
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage

from app.clients.minio_utils import get_minio_client
from app.conf.lm_config import lm_config
from app.conf.minio_config import minio_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.lm.llm_utils import get_llm_client
from app.utils.task_utils import add_done_task, add_running_task

# Markdown 图片语法：![alt](path)
_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
# 生成描述时截取的上下文长度，前后各取这么多字符
CONTEXT_CHARS = 200
# 常见图片扩展名 → MIME 类型，MinIO 需要正确的 content-type 才能让浏览器直接预览
_CONTENT_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
}


def _public_url(object_name: str) -> str:
    """拼接 MinIO 的公网访问地址。桶已配置匿名只读，可直接用裸 URL。"""
    scheme = "https" if minio_config.minio_secure else "http"
    return f"{scheme}://{minio_config.endpoint}/{minio_config.bucket_name}/{object_name}"


def _upload_image(client, local_path: Path, object_name: str) -> str | None:
    """上传单张图片，返回公网 URL；失败返回 None。"""
    try:
        content_type = _CONTENT_TYPES.get(local_path.suffix.lower(), "application/octet-stream")
        client.fput_object(
            bucket_name=minio_config.bucket_name,
            object_name=object_name,
            file_path=str(local_path),
            content_type=content_type,
        )
        return _public_url(object_name)
    except Exception as e:
        logger.warning(f"图片上传失败 [{local_path.name}]：{e}")
        return None


def _describe_image(image_url: str, file_title: str, before: str, after: str) -> str:
    """调用多模态模型生成图片描述；失败返回空串，由调用方决定如何降级。"""
    try:
        prompt = load_prompt(
            "image_summary", root_folder=file_title, image_content=(before, after)
        )
        # 必须用视觉模型（VL_MODEL），默认的纯文本模型收到 image_url 会直接报错
        # 走 OpenAI 兼容的多模态消息格式：文本 + 图片 URL 混合输入
        vl_client = get_llm_client(model=lm_config.lv_model)
        response = vl_client.invoke([
            HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ])
        ])
        return (response.content or "").strip().replace("\n", " ")
    except Exception as e:
        logger.warning(f"图片描述生成失败 [{image_url}]：{e}")
        return ""


def node_md_img(state: ImportGraphState) -> ImportGraphState:
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}] 开始执行，文件：{state.get('file_title')}")
    add_running_task(state["task_id"], function_name)

    try:
        md_content = state.get("md_content") or ""
        md_path = state.get("md_path")

        # 走 MD 直读分支时 md_content 尚未填充，这里补读一次
        if not md_content and md_path and Path(md_path).exists():
            md_content = Path(md_path).read_text(encoding="utf-8")
            state["md_content"] = md_content

        if not md_content.strip():
            logger.error(f"[{function_name}] md_content 为空，跳过图片处理")
            return state

        matches = list(_IMAGE.finditer(md_content))
        if not matches:
            logger.info(f"[{function_name}] 文档中没有图片，跳过")
            return state

        client = get_minio_client()
        if client is None:
            # MinIO 不可用时保留原始链接，不影响正文入库
            logger.warning(f"[{function_name}] MinIO 客户端不可用，保留原始图片链接")
            return state

        base_dir = Path(md_path).parent if md_path else Path(state.get("local_dir") or ".")
        file_title = state.get("file_title") or "document"
        img_dir = (minio_config.minio_img_dir or "/images").strip("/")

        uploaded = described = skipped = 0
        replacements: list[tuple[str, str]] = []

        for m in matches:
            alt_text, raw_path = m.group(1), m.group(2).strip()

            # 已经是外链的图片不重复上传
            if raw_path.startswith(("http://", "https://")):
                skipped += 1
                continue

            local_path = (base_dir / raw_path).resolve()
            if not local_path.exists():
                logger.warning(f"[{function_name}] 图片文件不存在，跳过：{local_path}")
                skipped += 1
                continue

            object_name = f"{img_dir}/{file_title}/{local_path.name}"
            url = _upload_image(client, local_path, object_name)
            if not url:
                skipped += 1
                continue
            uploaded += 1

            # 取图片在原文中的上下文，供多模态模型参考
            before = md_content[max(0, m.start() - CONTEXT_CHARS):m.start()].strip()
            after = md_content[m.end():m.end() + CONTEXT_CHARS].strip()
            summary = _describe_image(url, file_title, before, after)
            if summary:
                described += 1
            # 描述写进 alt 文本：切分节点会把图片语法从正文剔除，
            # 但描述本身已随段落文本保留下来参与向量化
            new_alt = summary or alt_text or local_path.stem
            replacements.append((m.group(0), f"![{new_alt}]({url})"))

        for old, new in replacements:
            md_content = md_content.replace(old, new, 1)
        state["md_content"] = md_content

        # 回写 Markdown 文件，便于人工核对替换结果
        if md_path:
            try:
                Path(md_path).write_text(md_content, encoding="utf-8")
            except Exception as e:
                logger.warning(f"[{function_name}] Markdown 回写失败（不影响后续流程）：{e}")

        logger.success(
            f"[{function_name}] 图片处理完成：共 {len(matches)} 张，"
            f"上传 {uploaded} 张，生成描述 {described} 条，跳过 {skipped} 张"
        )

    except Exception as e:
        # 图片是增强项，失败不阻断导入——正文照常进入后续切分与入库
        logger.error(f"[{function_name}] 图片处理异常，保留原文继续流程：{e}", exc_info=True)
    finally:
        add_done_task(state["task_id"], function_name)

    return state


if __name__ == "__main__":
    from app.import_process.agent.state import create_default_state

    st = create_default_state(
        task_id="test_md_img",
        file_title="demo",
        md_content="# 标题\n背面接口说明如下。\n![](images/panel.png)\n请按图示连接电源。",
    )
    node_md_img(st)
    logger.info(st["md_content"])
