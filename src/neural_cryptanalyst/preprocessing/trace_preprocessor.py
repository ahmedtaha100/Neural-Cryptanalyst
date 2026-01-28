import numpy as np
from scipy import signal, stats
from scipy.signal import savgol_filter, detrend, sosfiltfilt, butter
from scipy.ndimage import median_filter
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings
from typing import Optional, Tuple, Union, List, Dict, Any
from joblib import Parallel, delayed

class TracePreprocessor:

    def __init__(self, sampling_rate: float = 1e9, dtype: np.dtype = np.float32,
                 n_jobs: int = 1, seed: Optional[int] = None):
        if sampling_rate <= 0:
            raise ValueError(f"Sampling rate must be positive, got {sampling_rate}")

        self.sampling_rate = sampling_rate
        self.dtype = dtype
        self.n_jobs = n_jobs if n_jobs > 0 else -1
        self.reference_trace = None
        self.scaler = None
        self._fitted = False
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def remove_dc_offset(self, traces: np.ndarray) -> np.ndarray:
        return traces - np.mean(traces, axis=1, keepdims=True)

    def align_traces_correlation(self, traces: np.ndarray,
                                 reference_trace: Optional[np.ndarray] = None,
                                 max_shift: Optional[int] = None,
                                 upsampling_factor: int = 1,
                                 use_fft: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if reference_trace is None:
            reference_trace = np.median(traces, axis=0)

        n_traces, n_samples = traces.shape
        if max_shift is None:
            max_shift = min(n_samples // 10, 1000)
        elif max_shift >= n_samples:
            raise ValueError(f"max_shift ({max_shift}) must be less than trace length ({n_samples})")

        aligned_traces = np.empty_like(traces)
        shifts = np.zeros(n_traces, dtype=np.float32 if upsampling_factor > 1 else np.int32)

        if upsampling_factor > 1:
            reference_upsampled = signal.resample(reference_trace, len(reference_trace) * upsampling_factor)

        def align_single_trace(i: int) -> Tuple[np.ndarray, Union[float, int]]:
            if upsampling_factor > 1:
                trace_upsampled = signal.resample(traces[i], len(traces[i]) * upsampling_factor)

                if use_fft:
                    correlation = signal.fftconvolve(trace_upsampled, reference_upsampled[::-1], mode='same')
                else:
                    correlation = signal.correlate(trace_upsampled, reference_upsampled, mode='same')

                max_idx = np.argmax(np.abs(correlation))
                shift = (max_idx - len(trace_upsampled) // 2) / upsampling_factor

                if abs(shift) > max_shift:
                    return traces[i], 0.0

                t_old = np.arange(len(traces[i]))
                t_new = t_old + shift
                aligned = np.interp(t_old, t_new, traces[i], left=traces[i][0], right=traces[i][-1])

            else:
                if use_fft:
                    correlation = signal.fftconvolve(traces[i], reference_trace[::-1], mode='same')
                else:
                    correlation = signal.correlate(traces[i], reference_trace, mode='same')

                max_idx = np.argmax(np.abs(correlation))
                shift = max_idx - n_samples // 2

                if abs(shift) > max_shift:
                    return traces[i], 0

                if shift > 0:
                    aligned = np.pad(traces[i], (0, shift), mode='edge')[shift:]
                elif shift < 0:
                    aligned = np.pad(traces[i], (-shift, 0), mode='edge')[:shift]
                else:
                    aligned = traces[i]

            return aligned[:n_samples], shift

        if self.n_jobs == 1:
            for i in range(n_traces):
                aligned_traces[i], shifts[i] = align_single_trace(i)
        else:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(align_single_trace)(i) for i in range(n_traces)
            )
            for i, (aligned, shift) in enumerate(results):
                aligned_traces[i] = aligned
                shifts[i] = shift

        return aligned_traces, shifts

    def apply_filter(self, traces: np.ndarray, filter_type: str = 'lowpass',
                     cutoff: Union[float, Tuple[float, float]] = 0.4, order: int = 5) -> np.ndarray:
        nyquist = 0.5 * self.sampling_rate

        if filter_type in ['lowpass', 'highpass']:
            if isinstance(cutoff, (int, float)):
                normal_cutoff = cutoff if cutoff <= 0.5 else cutoff / nyquist
            else:
                normal_cutoff = cutoff[0] if cutoff[0] <= 0.5 else cutoff[0] / nyquist
        else:
            normal_cutoff = [c if c <= 0.5 else c / nyquist for c in cutoff]

        sos = butter(order, normal_cutoff, btype=filter_type, output='sos')

        filtered = np.empty_like(traces)
        for i in range(len(traces)):
            filtered[i] = sosfiltfilt(sos, traces[i])

        return filtered

    def fit(self, traces: np.ndarray, method: str = 'robust',
            compute_reference: bool = True) -> 'TracePreprocessor':
        traces = self._validate_input(traces)
        traces = self.remove_dc_offset(traces)

        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        self.scaler.fit(traces)

        if compute_reference:
            self.reference_trace = np.median(traces, axis=0).astype(self.dtype)

        self._fitted = True
        return self

    def _validate_input(self, traces: np.ndarray) -> np.ndarray:
        if not isinstance(traces, np.ndarray):
            traces = np.array(traces)

        if traces.ndim == 1:
            traces = traces.reshape(1, -1)
        elif traces.ndim != 2:
            raise ValueError(f"Expected 1D or 2D array, got {traces.ndim}D")

        if traces.size == 0:
            raise ValueError("Empty trace array provided")

        if np.any(np.isnan(traces)) or np.any(np.isinf(traces)):
            warnings.warn("Input contains NaN or inf values, replacing with 0")
            traces = np.nan_to_num(traces, nan=0.0, posinf=0.0, neginf=0.0)

        return traces.astype(self.dtype)

    def detrend_traces(self, traces: np.ndarray, type: str = 'linear') -> np.ndarray:
        if type not in ['linear', 'constant']:
            raise ValueError(f"Detrend type must be 'linear' or 'constant', got '{type}'")

        detrended = np.empty_like(traces)
        for i in range(len(traces)):
            detrended[i] = detrend(traces[i], type=type)

        return detrended

    def standardize(self, traces: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ValueError("Preprocessor must be fitted first")
        return self.scaler.transform(traces)

    def preprocess_traces(self, traces: np.ndarray,
                         pipeline: Optional[List[str]] = None) -> np.ndarray:
        traces = self._validate_input(traces)

        if pipeline is None:
            pipeline = ['dc_offset', 'detrend', 'align', 'filter', 'standardize']

        for step in pipeline:
            if step == 'dc_offset':
                traces = self.remove_dc_offset(traces)
            elif step == 'detrend':
                traces = self.detrend_traces(traces)
            elif step == 'align':
                ref_trace = self.reference_trace if self._fitted else None
                traces, _ = self.align_traces_correlation(traces, reference_trace=ref_trace)
            elif step == 'filter':
                traces = self.apply_filter(traces)
            elif step == 'standardize':
                if self._fitted:
                    traces = self.standardize(traces)
                else:
                    warnings.warn("Preprocessor not fitted, skipping standardization")

        return traces

    def detect_and_remove_outliers(self, traces: np.ndarray, threshold: float = 3.0, method: str = 'mad') -> Tuple[np.ndarray, np.ndarray]:
        if threshold <= 0:
            raise ValueError(f"Threshold must be positive, got {threshold}")
        if method not in ['mad', 'std']:
            raise ValueError(f"Method must be 'mad' or 'std', got {method}")
        if method == 'mad':
            median_trace = np.median(traces, axis=0)
            distances = np.sqrt(np.sum((traces - median_trace)**2, axis=1))
            mad = np.median(np.abs(distances - np.median(distances)))
            if mad < 1e-10:
                warnings.warn("MAD is near zero, using standard deviation method")
                method = 'std'
            else:
                mad_scale = 1.4826
                threshold_val = np.median(distances) + threshold * mad * mad_scale
                outlier_mask = distances > threshold_val
        if method == 'std':
            mean_trace = np.mean(traces, axis=0)
            distances = np.sqrt(np.sum((traces - mean_trace)**2, axis=1))
            threshold_val = np.mean(distances) + threshold * np.std(distances)
            outlier_mask = distances > threshold_val
        clean_traces = traces[~outlier_mask]
        if len(clean_traces) < len(traces) * 0.1:
            warnings.warn(f"Would remove {np.sum(outlier_mask)} outliers ({np.mean(outlier_mask)*100:.1f}%), returning original traces")
            return traces, np.zeros(len(traces), dtype=bool)
        return clean_traces, outlier_mask

    def align_traces_dtw(self, traces: np.ndarray, reference_trace: Optional[np.ndarray] = None, window_size: int = 100) -> Tuple[np.ndarray, List]:
        try:
            from dtaidistance import dtw
        except ImportError:
            raise ImportError("DTW alignment requires dtaidistance package: pip install dtaidistance")
        if reference_trace is None:
            if not self._fitted:
                raise ValueError("Preprocessor must be fitted or reference trace provided")
            reference_trace = self.reference_trace
        n_traces, n_samples = traces.shape
        if window_size > n_samples // 2:
            warnings.warn(f"DTW window size {window_size} is large for trace length {n_samples}")
            window_size = n_samples // 4
        aligned_traces = np.empty_like(traces)
        warp_paths = []
        for i in range(n_traces):
            distance, paths = dtw.warping_paths(traces[i], reference_trace, window=window_size, use_c=True)
            path = dtw.best_path(paths)
            path_array = np.array(path)
            aligned = np.interp(np.arange(n_samples), path_array[:, 0], traces[i][path_array[:, 0]])
            aligned_traces[i] = aligned
            warp_paths.append(path)
        return aligned_traces, warp_paths

    def align_traces_peak(self, traces: np.ndarray, peak_window: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        if peak_window[0] < 0 or peak_window[1] > traces.shape[1] or peak_window[0] >= peak_window[1]:
            raise ValueError(f"Invalid peak window: {peak_window}")
        n_traces, n_samples = traces.shape
        aligned_traces = np.empty_like(traces)
        shifts = np.zeros(n_traces, dtype=np.int32)
        start, end = peak_window
        reference_peak = start + np.argmax(np.mean(np.abs(traces[:, start:end]), axis=0))
        for i in range(n_traces):
            peak_pos = start + np.argmax(np.abs(traces[i, start:end]))
            shift = reference_peak - peak_pos
            shifts[i] = shift
            if abs(shift) < n_samples:
                if shift > 0:
                    aligned_traces[i, shift:] = traces[i, :-shift] if shift < n_samples else traces[i][-1]
                    aligned_traces[i, :shift] = traces[i, 0]
                elif shift < 0:
                    aligned_traces[i, :shift] = traces[i, -shift:] if -shift < n_samples else traces[i][0]
                    aligned_traces[i, shift:] = traces[i, -1]
                else:
                    aligned_traces[i] = traces[i]
            else:
                aligned_traces[i] = traces[i]
                shifts[i] = 0
        return aligned_traces, shifts

    def _apply_savitzky_golay(self, traces: np.ndarray, window_length: int = 51, polyorder: int = 3) -> np.ndarray:
        if polyorder >= window_length:
            raise ValueError("Polynomial order must be less than window length")
        if window_length % 2 == 0:
            raise ValueError("Window length must be odd")
        if window_length < 3:
            raise ValueError("Window length must be at least 3")
        filtered = np.empty_like(traces)
        for i in range(len(traces)):
            trace_len = len(traces[i])
            if window_length >= trace_len:
                actual_window = trace_len - 1 if trace_len % 2 == 0 else trace_len
                if actual_window % 2 == 0:
                    actual_window -= 1
                actual_window = max(3, actual_window)
                actual_poly = min(polyorder, actual_window - 1)
            else:
                actual_window = window_length
                actual_poly = polyorder
            if actual_window <= trace_len and actual_poly < actual_window:
                try:
                    filtered[i] = savgol_filter(traces[i], actual_window, actual_poly)
                except Exception as e:
                    warnings.warn(f"Savitzky-Golay filter failed for trace {i}: {e}")
                    filtered[i] = traces[i]
            else:
                filtered[i] = traces[i]
        return filtered

    def _apply_fft_filter(self, traces: np.ndarray, cutoff_low: Optional[float] = None, cutoff_high: Optional[float] = None) -> np.ndarray:
        if cutoff_low is None and cutoff_high is None:
            raise ValueError("At least one cutoff frequency must be specified")
        filtered = np.empty_like(traces)
        n_samples = traces.shape[1]
        freqs = np.fft.fftfreq(n_samples, 1/self.sampling_rate)
        for i in range(len(traces)):
            fft = np.fft.fft(traces[i])
            if cutoff_low is not None:
                fft[np.abs(freqs) < cutoff_low] = 0
            if cutoff_high is not None:
                fft[np.abs(freqs) > cutoff_high] = 0
            filtered[i] = np.real(np.fft.ifft(fft))
        return filtered

    def resample_traces(self, traces: np.ndarray, target_length: int, method: str = 'fourier', window: Optional[str] = None) -> np.ndarray:
        if target_length <= 0:
            raise ValueError(f"Target length must be positive, got {target_length}")
        if method not in ['fourier', 'linear', 'cubic']:
            raise ValueError(f"Method must be 'fourier', 'linear', or 'cubic', got '{method}'")
        n_traces = len(traces)
        resampled = np.empty((n_traces, target_length), dtype=self.dtype)
        for i in range(n_traces):
            if method == 'fourier':
                if window is not None:
                    resampled[i] = signal.resample(traces[i], target_length, window=window)
                else:
                    resampled[i] = signal.resample(traces[i], target_length)
            elif method == 'linear':
                x_old = np.linspace(0, 1, len(traces[i]))
                x_new = np.linspace(0, 1, target_length)
                resampled[i] = np.interp(x_new, x_old, traces[i])
            elif method == 'cubic':
                from scipy.interpolate import interp1d
                x_old = np.linspace(0, 1, len(traces[i]))
                x_new = np.linspace(0, 1, target_length)
                f = interp1d(x_old, traces[i], kind='cubic', fill_value='extrapolate')
                resampled[i] = f(x_new)
        return resampled
