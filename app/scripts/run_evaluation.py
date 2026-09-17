"""知识库问答端到端评测脚本。

问答类系统没有唯一正确的答案字符串，所以不做全文比对，而是分两层打分：

**检索命中率**：正确的章节是否进入了精排后的上下文。
**答案关键事实覆盖率**：答案中是否包含绕不开的关键信息。

两层分开统计是刻意的。一道题答错，可能是检索压根没找到对的片段，
也可能是找到了但模型没用好——这两种失败的优化方向完全不同
（前者调切分和检索策略，后者调提示词），混进一个总分里就看不出该改哪儿。

用法：
    python -m app.scripts.run_evaluation
    python -m app.scripts.run_evaluation --level L1 L2
    python -m app.scripts.run_evaluation -c evaluation/golden_set.yaml --concurrency 2
"""

import argparse
import asyncio
import json
import re
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

from app.core.logger import logger
from app.query_process.agent.main_graph import answer_question
from app.query_process.agent.nodes.node_answer_output import NO_CONTEXT_ANSWER
from app.utils.path_util import PROJECT_ROOT

# 答案关键事实覆盖率达到该比例即判为答对。
# 不要求 100%：同一事实有多种表述，强制全命中会把正确答案误判为错。
KEYWORD_PASS_RATIO = 0.6

# 判定「模型在拒答」的特征词。除了系统内置的固定话术，
# 模型自己组织的拒答措辞也要能识别出来。
REFUSAL_MARKERS = (
    "没有检索到", "无法作答", "未提及", "没有提到", "手册中未", "无相关",
    "抱歉", "不包含", "未找到", "没有找到", "无法回答",
)


# 千分位分隔的数字，如 336,000 或 1,234,567
_THOUSAND_SEP = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


def normalize_numbers(text: str) -> str:
    """去掉数字中的千分位逗号，便于与评测集里的裸数字比对。

    模型习惯把大数写成 336,000 以便阅读，而评测集里写的是 336000，
    直接做子串匹配会把完全正确的答案判成错——这是测量工具的缺陷，
    不修的话准确率会被系统性低估。
    """
    return _THOUSAND_SEP.sub("", text)


def is_refusal(answer: str) -> bool:
    """判断答案是否为拒答。"""
    if not answer or answer.strip() == NO_CONTEXT_ANSWER.strip():
        return True
    return any(marker in answer for marker in REFUSAL_MARKERS)


def score_retrieval(chunks: list, expect_sections: list) -> Optional[bool]:
    """检索是否命中期望章节。未声明期望章节时返回 None，不计入统计。"""
    if not expect_sections:
        return None
    hit_titles = set()
    for c in chunks:
        hit_titles.add((c.get("title") or "").strip())
        hit_titles.add((c.get("parent_title") or "").strip())
    # 用包含而非相等：手册标题常带编号前缀（"2.1 接线"），
    # 精确匹配会因为编号差异大面积误判
    return any(
        any(exp in t or t in exp for t in hit_titles if t)
        for exp in expect_sections
    )


def score_keywords(answer: str, expect: list, forbid: list) -> tuple[Optional[float], list]:
    """返回 (关键事实覆盖率, 命中的禁用词)。"""
    normalized = normalize_numbers(answer)
    violated = [k for k in (forbid or []) if k in answer or k in normalized]
    if not expect:
        return None, violated
    hit = sum(1 for k in expect if k in answer or k in normalized)
    return hit / len(expect), violated


async def evaluate_one(case: dict, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        record: dict[str, Any] = {
            "id": case["id"],
            "level": case["level"],
            "question": case["question"],
            "expect_refusal": bool(case.get("expect_refusal", False)),
            "answer": "",
            "retrieval_hit": None,
            "keyword_ratio": None,
            "passed": False,
            "reason": "",
            "latency_ms": None,
        }

        started = time.perf_counter()
        try:
            # answer_question 是同步调用，放进线程池避免阻塞事件循环，
            # 否则设置并发也不会真的并发
            result = await asyncio.to_thread(
                answer_question, case["question"], None, f"eval_{case['id']}"
            )
        except Exception as exc:
            record["latency_ms"] = round((time.perf_counter() - started) * 1000)
            record["reason"] = f"链路异常: {type(exc).__name__}: {exc}"[:200]
            return record

        record["latency_ms"] = round((time.perf_counter() - started) * 1000)
        answer = result.get("answer", "")
        chunks = result.get("chunks", [])
        record["answer"] = answer[:500]
        record["chunk_count"] = len(chunks)

        # --- 拒答题：只看有没有守住底线 ---
        if record["expect_refusal"]:
            if is_refusal(answer):
                record["passed"] = True
                record["reason"] = "正确拒答"
            else:
                record["reason"] = f"应拒答却给出了内容（{answer[:60]}...）"
            return record

        # --- 常规题：检索命中 + 关键事实覆盖 ---
        record["retrieval_hit"] = score_retrieval(chunks, case.get("expect_sections") or [])
        ratio, violated = score_keywords(
            answer, case.get("expect_keywords") or [], case.get("forbid_keywords") or []
        )
        record["keyword_ratio"] = ratio

        if violated:
            record["reason"] = f"出现禁用表述（疑似答反）：{violated}"
            return record
        if is_refusal(answer):
            record["reason"] = "本应能答出却拒答了，检查检索是否召回为空"
            return record
        if ratio is not None and ratio < KEYWORD_PASS_RATIO:
            missing = [k for k in case.get("expect_keywords", []) if k not in answer]
            record["reason"] = f"关键事实覆盖不足 {ratio:.0%}，缺失：{missing}"
            return record

        record["passed"] = True
        record["reason"] = (
            f"通过（关键事实 {ratio:.0%}，检索"
            f"{'命中' if record['retrieval_hit'] else '未命中期望章节'}）"
            if ratio is not None else "通过"
        )
        return record


def summarize(records: list[dict]) -> dict:
    total = len(records)
    passed = sum(r["passed"] for r in records)
    latencies = sorted(r["latency_ms"] for r in records if r["latency_ms"] is not None)

    retrieval = [r["retrieval_hit"] for r in records if r["retrieval_hit"] is not None]
    ratios = [r["keyword_ratio"] for r in records if r["keyword_ratio"] is not None]
    refusal = [r for r in records if r["expect_refusal"]]

    by_level: dict[str, dict] = {}
    for r in records:
        b = by_level.setdefault(r["level"], {"total": 0, "passed": 0})
        b["total"] += 1
        b["passed"] += int(r["passed"])

    def pct(n, d) -> Optional[float]:
        """分母为 0 时返回 None——没有样本不等于准确率为零。"""
        return round(n / d * 100, 1) if d else None

    def percentile(data, p) -> Optional[int]:
        if not data:
            return None
        return data[min(int(len(data) * p), len(data) - 1)]

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total": total,
        "passed": passed,
        "accuracy_pct": pct(passed, total),
        "retrieval_hit_pct": pct(sum(retrieval), len(retrieval)),
        "avg_keyword_coverage_pct": round(statistics.mean(ratios) * 100, 1) if ratios else None,
        "refusal_accuracy_pct": pct(sum(r["passed"] for r in refusal), len(refusal)),
        "by_level": {
            lv: {**v, "accuracy_pct": pct(v["passed"], v["total"])}
            for lv, v in sorted(by_level.items())
        },
        "latency_ms": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "mean": round(statistics.mean(latencies)) if latencies else None,
        },
    }


def fmt(v: Optional[float]) -> str:
    return "N/A" if v is None else f"{v}%"


def render_markdown(summary: dict, records: list[dict]) -> str:
    lines = [
        "## 评测结果", "",
        f"- 评测时间：{summary['generated_at']}",
        f"- 题目总数：{summary['total']}", "",
        "| 指标 | 数值 |", "| --- | --- |",
        f"| 总体准确率 | {fmt(summary['accuracy_pct'])} "
        f"({summary['passed']}/{summary['total']}) |",
        f"| 检索命中率 | {fmt(summary['retrieval_hit_pct'])} |",
        f"| 关键事实平均覆盖率 | {fmt(summary['avg_keyword_coverage_pct'])} |",
        f"| 拒答准确率 | {fmt(summary['refusal_accuracy_pct'])} |",
        f"| 端到端延迟 P50 | {summary['latency_ms']['p50']} ms |",
        f"| 端到端延迟 P95 | {summary['latency_ms']['p95']} ms |",
        "", "### 分层级准确率", "",
        "| 层级 | 通过 / 总数 | 准确率 |", "| --- | --- | --- |",
    ]
    for lv, v in summary["by_level"].items():
        lines.append(f"| {lv} | {v['passed']} / {v['total']} | {fmt(v['accuracy_pct'])} |")

    failed = [r for r in records if not r["passed"]]
    if failed:
        lines += ["", "### 未通过用例", "", "| 编号 | 问题 | 原因 |", "| --- | --- | --- |"]
        for r in failed:
            reason = r["reason"].replace("|", "\\|").replace("\n", " ")[:120]
            lines.append(f"| {r['id']} | {r['question']} | {reason} |")
    return "\n".join(lines) + "\n"


async def main(config_path: Path, levels: Optional[list], concurrency: int) -> None:
    cases = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if levels:
        cases = [c for c in cases if c["level"] in levels]
    if not cases:
        raise SystemExit("评测集为空，请检查 -c 路径或 --level 过滤条件")

    logger.info(f"开始评测：{len(cases)} 道题，并发度 {concurrency}")
    semaphore = asyncio.Semaphore(concurrency)
    records = await asyncio.gather(*(evaluate_one(c, semaphore) for c in cases))
    records = sorted(records, key=lambda r: r["id"])

    for r in records:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"  [{mark}] {r['id']:<7} {r['latency_ms']:>6}ms  {r['question']}")
        if not r["passed"]:
            print(f"          -> {r['reason']}")

    summary = summarize(records)
    print("\n" + "=" * 66)
    print(f"总体准确率        {fmt(summary['accuracy_pct'])} "
          f"({summary['passed']}/{summary['total']})")
    print(f"检索命中率        {fmt(summary['retrieval_hit_pct'])}")
    print(f"关键事实覆盖率    {fmt(summary['avg_keyword_coverage_pct'])}")
    print(f"拒答准确率        {fmt(summary['refusal_accuracy_pct'])}")
    print(f"延迟 P50 / P95    {summary['latency_ms']['p50']} / "
          f"{summary['latency_ms']['p95']} ms")
    print("=" * 66)

    reports_dir = PROJECT_ROOT / "evaluation" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (reports_dir / f"report_{stamp}.json").write_text(
        json.dumps({"summary": summary, "records": records},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (reports_dir / f"report_{stamp}.md").write_text(
        render_markdown(summary, records), encoding="utf-8")
    print(f"\n完整明细 -> {reports_dir / f'report_{stamp}.json'}")
    print(f"README 片段 -> {reports_dir / f'report_{stamp}.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="知识库问答端到端评测")
    parser.add_argument("-c", "--conf", default="evaluation/golden_set.yaml")
    parser.add_argument("--level", nargs="*", default=None, help="只跑指定层级，如 --level L1 L2")
    parser.add_argument("--concurrency", type=int, default=2,
                        help="并发题目数，受 LLM 限流与本地模型显存影响，默认 2")
    args = parser.parse_args()

    asyncio.run(main(Path(args.conf), args.level, args.concurrency))
