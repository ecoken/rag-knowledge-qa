import threading

from FlagEmbedding import FlagReranker

from app.conf.reranker_config import reranker_config

_reranker_model = None
# 保护单例初始化的锁，原因同 embedding_utils：
# 无锁单例在并发下会让多个线程同时加载同一个模型，
# 后到的线程可能读到尚未完成设备迁移的半成品，
# 抛出 "Cannot copy out of meta tensor"，且只在并发时复现。
_reranker_lock = threading.Lock()
# 保护推理调用的锁，理由同 embedding_utils：
# 单例锁只保证模型被加载一次，多个线程仍会并发调用同一个实例。
# GPU + FP16 下并发推理会抛 "expected scalar type Half but found Float"，
# 而 CPU + FP32 时同样的代码不会报错。
_reranker_infer_lock = threading.Lock()


def rerank_scores(pairs, normalize: bool = True):
    """对 [[查询, 文档], ...] 批量打分，串行化推理调用。

    交叉编码器要对每个候选单独前向推理，是整条链路里最重的一步，
    但也正因如此它天然适合批量调用——一次传入全部候选，
    锁的持有时间与不加锁时的总推理时间相当，几乎不损失吞吐。
    """
    model = get_reranker_model()
    with _reranker_infer_lock:
        scores = model.compute_score(pairs, normalize=normalize)
    # 单条候选时部分版本返回标量而非列表，统一成列表
    return scores if isinstance(scores, list) else [scores]


def get_reranker_model():
    """获取重排模型单例。

    双重检查加锁：已初始化时直接返回、不进锁，避免每次调用都产生锁竞争；
    只有首次初始化才需要互斥。
    """
    global _reranker_model
    if _reranker_model is not None:
        return _reranker_model

    with _reranker_lock:
        # 等锁期间可能已被其他线程完成初始化
        if _reranker_model is None:
            _reranker_model = FlagReranker(
                model_name_or_path=reranker_config.bge_reranker_large,
                device=reranker_config.bge_reranker_device,
                use_fp16=reranker_config.bge_reranker_fp16,
            )
    return _reranker_model
