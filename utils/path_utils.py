import os
import sys

def get_project_root() -> str:
    """
    Returns the absolute path to the project root directory.
    Assumes this file is located at <project_root>/utils/path_utils.py.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def resolve_path(rel_or_abs: str) -> str:
    """
    Resolves a path to an absolute path.
    If the path is already absolute, it is returned as is.
    If it is relative, it is resolved relative to the project root directory.
    """
    if not rel_or_abs:
        return ""
    if os.path.isabs(rel_or_abs):
        return os.path.normpath(rel_or_abs)
    # Strip leading dots and separators to resolve relative to root
    cleaned = rel_or_abs.lstrip('.').lstrip('/').lstrip('\\')
    return os.path.normpath(os.path.join(get_project_root(), cleaned))


def resolve_num_workers(configured: int) -> int:
    """解析 DataLoader num_workers；Windows 下多进程易卡死，强制为 0。"""
    if sys.platform == 'win32':
        return 0
    return configured


def is_rank0() -> bool:
    """单卡或分布式 rank 0 时返回 True，用于抑制多卡重复日志。"""
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"]) == 0
    if "RANK" in os.environ:
        return int(os.environ["RANK"]) == 0
    return True


def log_rank0(*args, **kwargs):
    """仅在 rank 0 进程打印。"""
    if is_rank0():
        print(*args, **kwargs)


def configure_dist_process_logging():
    """非 rank 0 进程压低第三方库日志与进度条（需在 LOCAL_RANK 就绪后调用）。"""
    if is_rank0():
        return
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TQDM_DISABLE", "1")
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("accelerate").setLevel(logging.ERROR)
    try:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    except ImportError:
        pass
    try:
        from transformers.utils.logging import disable_progress_bar
        disable_progress_bar()
    except ImportError:
        pass
    try:
        from transformers.utils import logging as tf_logging
        tf_logging.set_verbosity_error()
    except ImportError:
        pass
