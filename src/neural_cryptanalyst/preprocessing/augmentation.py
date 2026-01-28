import numpy as np
from typing import Optional, Tuple, List

class TraceAugmenter:

    def __init__(self, noise_level: float = 0.05, shift_range: int = 50,
                 scale_range: Tuple[float, float] = (0.9, 1.1),
                 random_state: Optional[int] = None):
        self.noise_level = noise_level
        self.shift_range = shift_range
        self.scale_range = scale_range
        self.rng = np.random.RandomState(random_state)

    def add_gaussian_noise(self, traces: np.ndarray,
                           noise_level: Optional[float] = None) -> np.ndarray:
        if noise_level is None:
            noise_level = self.noise_level

        augmented = np.zeros_like(traces)
        for i in range(len(traces)):
            trace_std = np.std(traces[i]) + 1e-10
            noise = self.rng.normal(0, noise_level * trace_std, traces[i].shape)
            augmented[i] = traces[i] + noise

        return augmented

    def add_synthetic_noise(self, traces: np.ndarray,
                             noise_type: str = 'pink') -> np.ndarray:
        import scipy.signal

        augmented = np.zeros_like(traces)
        n_samples = traces.shape[1]

        for i in range(len(traces)):
            if noise_type == 'white':
                noise = self.rng.randn(n_samples)
            elif noise_type == 'pink':
                white = self.rng.randn(n_samples)
                b, a = scipy.signal.butter(1, 0.1)
                noise = scipy.signal.filtfilt(b, a, white)
            elif noise_type == 'brown':
                white = self.rng.randn(n_samples)
                noise = np.cumsum(white) / np.sqrt(n_samples)
            else:
                raise ValueError(f"Unknown noise type: {noise_type}")

            noise_power = np.std(noise)
            signal_power = np.std(traces[i])
            noise = noise * (self.noise_level * signal_power / (noise_power + 1e-10))

            augmented[i] = traces[i] + noise

        return augmented

    def random_shift(self, traces: np.ndarray,
                     shift_range: Optional[int] = None) -> np.ndarray:
        if shift_range is None:
            shift_range = self.shift_range

        augmented = np.zeros_like(traces)
        shifts = self.rng.randint(-shift_range, shift_range + 1, size=len(traces))

        for i, (trace, shift) in enumerate(zip(traces, shifts)):
            augmented[i] = np.roll(trace, shift)

        return augmented

    def random_scale(self, traces: np.ndarray,
                     scale_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
        if scale_range is None:
            scale_range = self.scale_range

        scales = self.rng.uniform(scale_range[0], scale_range[1],
                                 size=(len(traces), 1))
        return traces * scales

    def augment_batch(self, traces: np.ndarray, labels: np.ndarray,
                      augmentations: List[str] = ['noise', 'shift', 'scale'],
                      num_augmented: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        if len(traces) != len(labels):
            raise ValueError("Traces and labels must have same length")

        all_traces = [traces]
        all_labels = [labels]

        for _ in range(num_augmented):
            augmented = traces.copy()

            for aug_type in augmentations:
                if aug_type == 'noise':
                    augmented = self.add_gaussian_noise(augmented)
                elif aug_type == 'shift':
                    augmented = self.random_shift(augmented)
                elif aug_type == 'scale':
                    augmented = self.random_scale(augmented)

            all_traces.append(augmented)
            all_labels.append(labels.copy())

        return np.vstack(all_traces), np.hstack(all_labels)
