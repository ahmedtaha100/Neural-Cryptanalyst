import numpy as np
from typing import Tuple, Union

class HidingCountermeasure:

    def add_random_delays(self, traces: np.ndarray, max_delay: int = 100,
                          return_delays: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        n_traces, trace_length = traces.shape
        new_length = trace_length + max_delay

        augmented = np.zeros((n_traces, new_length), dtype=traces.dtype)
        delays = np.random.randint(0, max_delay + 1, size=n_traces)

        for i, (trace, delay) in enumerate(zip(traces, delays)):
            augmented[i, delay:delay + trace_length] = trace

        if return_delays:
            return augmented, delays
        return augmented

    def add_dummy_operations(self, traces: np.ndarray, num_dummies: int = 5,
                             dummy_pattern: str = 'random') -> np.ndarray:
        n_traces, trace_length = traces.shape
        window_size = trace_length // (num_dummies + 1)

        augmented = traces.copy()

        for i in range(n_traces):
            for j in range(num_dummies):
                insert_pos = np.random.randint(0, trace_length - window_size)

                if dummy_pattern == 'random':
                    dummy = np.random.randn(window_size) * np.std(traces[i])
                elif dummy_pattern == 'constant':
                    dummy = np.ones(window_size) * np.mean(traces[i])
                elif dummy_pattern == 'copy':
                    copy_pos = np.random.randint(0, trace_length - window_size)
                    dummy = traces[i, copy_pos:copy_pos + window_size]
                else:
                    raise ValueError(f"Unknown dummy pattern: {dummy_pattern}")

                alpha = np.random.uniform(0.3, 0.7)
                augmented[i, insert_pos:insert_pos + window_size] = (
                    alpha * dummy + (1 - alpha) * augmented[i, insert_pos:insert_pos + window_size]
                )

        return augmented

    def shuffle_operations(self, traces: np.ndarray, num_segments: int = 4) -> np.ndarray:
        n_traces, trace_length = traces.shape
        segment_length = trace_length // num_segments

        augmented = np.zeros_like(traces)

        for i in range(n_traces):
            segments = [
                traces[i, j * segment_length:(j + 1) * segment_length]
                for j in range(num_segments)
            ]

            order = np.random.permutation(num_segments)
            shuffled_segments = [segments[j] for j in order]

            for j, seg in enumerate(shuffled_segments):
                augmented[i, j * segment_length:(j + 1) * segment_length] = seg

            remainder = trace_length % num_segments
            if remainder > 0:
                augmented[i, -remainder:] = traces[i, -remainder:]

        return augmented
