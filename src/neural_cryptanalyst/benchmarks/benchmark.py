import time
import numpy as np

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class AttackBenchmark:

    def __init__(self):
        self.results = {}

    def benchmark_model(self, model, traces, labels, validation_split: float = 0.2):
        import tensorflow as tf

        start_time = time.time()
        start_memory = 0
        if PSUTIL_AVAILABLE:
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024

        if len(traces.shape) == 2:
            traces = traces.reshape(traces.shape[0], traces.shape[1], 1)

        if len(labels.shape) == 1:
            num_classes = getattr(model, 'num_classes', 256)
            labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)

        split_idx = int(len(traces) * (1 - validation_split))
        X_train, X_val = traces[:split_idx], traces[split_idx:]
        y_train, y_val = labels[:split_idx], labels[split_idx:]

        model.compile_model()
        history = model.train(X_train, y_train, X_val, y_val, epochs=10)

        train_time = time.time() - start_time
        memory_used = None
        if PSUTIL_AVAILABLE:
            memory_used = (
                psutil.Process().memory_info().rss / 1024 / 1024 - start_memory
            )

        gpu_memory = None
        if GPU_AVAILABLE:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_memory = gpus[0].memoryUsed

        return {
            "train_time": train_time,
            "memory_mb": memory_used,
            "gpu_memory_mb": gpu_memory,
            "final_accuracy": history.history.get("accuracy", [0])[-1],
        }

    def compare_models(self, models: dict, traces: np.ndarray, labels: np.ndarray,
                       validation_split: float = 0.2) -> dict:
        results = {}
        for name, model in models.items():
            print(f"Benchmarking {name}...")
            results[name] = self.benchmark_model(
                model, traces.copy(), labels.copy(), validation_split
            )
        return results
