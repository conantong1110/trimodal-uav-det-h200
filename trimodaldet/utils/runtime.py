"""
Runtime setup shared by every entry script: live logs, GPU math settings, watchdog.
"""
import sys

import torch

from .watchdog import install_log_watchdog


def setup_runtime(watchdog_minutes=None):
    """
    Prepare the process for a long GPU run.

    Args:
        watchdog_minutes: Passed to install_log_watchdog (None = env var or 40).
    """
    # Line-buffer so each log line reaches the terminal / log file immediately,
    # also when the output is piped through `tee` or written by a job scheduler.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except AttributeError:
            pass

    install_log_watchdog(watchdog_minutes)

    if torch.cuda.is_available():
        # TF32 matmul/conv and cuDNN autotuning: the standard speed settings for
        # recent NVIDIA data-center GPUs (H200 included) while staying in float32.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.total_memory / 1024 ** 3:.0f} GB) | "
              f"torch {torch.__version__} | CUDA {torch.version.cuda}")
    else:
        print("WARNING: CUDA is not available - running on CPU.")


__all__ = ['setup_runtime']
