import threading

from FlagEmbedding import FlagReranker

from app.conf.reranker_config import reranker_config

_reranker_model = None
# 保护单例初始化的锁，原因同 embedding_utils：
# 无锁单例在并发下会让多个线程同时加载同一个模型，
# 后到的线程可能读到尚未完成设备迁移的半成品，
# 抛出 "Cannot copy out of meta tensor"，且只在并发时复现。
_reranker_lock = threading.Lock()


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
