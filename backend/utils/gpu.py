"""TensorFlow GPU configuration helpers."""
from __future__ import annotations

import tensorflow as tf


def configure_gpu_memory_growth() -> list[str]:
    """Enable memory growth for available GPUs and return device names."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            # Memory growth must be configured before TensorFlow initializes GPUs.
            pass
    return [gpu.name for gpu in gpus]
