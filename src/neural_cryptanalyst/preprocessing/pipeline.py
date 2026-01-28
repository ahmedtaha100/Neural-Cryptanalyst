import h5py
import numpy as np
from typing import Optional, Union, Tuple, List
from .trace_preprocessor import TracePreprocessor
from .feature_selector import FeatureSelector
from .augmentation import TraceAugmenter

class TracePipeline:
    def __init__(self, preprocessor: TracePreprocessor, selector: FeatureSelector, augmenter: Optional[TraceAugmenter] = None):
        self.preprocessor = preprocessor
        self.selector = selector
        self.augmenter = augmenter

    def fit(self, traces: np.ndarray, labels: np.ndarray, preprocessing_steps: List[str] = None, feature_method: str = 'sost', num_features: int = 1000):
        processed = self.preprocessor.fit(traces).preprocess_traces(traces, preprocessing_steps)
        if feature_method == 'pca':
            self.selector.apply_pca(processed, n_components=num_features)
        else:
            self.selector.select_poi_sost(processed, labels, num_poi=num_features)
        return self

    def transform(self, traces: np.ndarray, labels: Optional[np.ndarray] = None, augment: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        processed = self.preprocessor.preprocess_traces(traces)
        transformed = self.selector.transform(processed)
        if augment and self.augmenter is not None and labels is not None:
            transformed, labels = self.augmenter.augment_batch(transformed, labels)
            return transformed, labels
        return transformed

    def process_large_dataset(self, filepath: str, batch_size: int = 1000,
                             output_file: Optional[str] = None,
                             compression: str = 'gzip', chunk_size: int = 32):
        import time
        from tqdm import tqdm

        stats = {
            'total_traces': 0,
            'processing_time': 0,
            'traces_per_second': 0,
            'output_file': output_file
        }

        start_time = time.time()

        with h5py.File(filepath, 'r') as f:
            if 'traces' not in f or 'labels' not in f:
                raise ValueError(f"File {filepath} must contain 'traces' and 'labels' datasets")

            n_traces = f['traces'].shape[0]
            n_samples = f['traces'].shape[1]
            stats['total_traces'] = n_traces

            if self.selector.selected_indices is None and not (hasattr(self.selector, 'pca') and self.selector.pca is not None):
                raise ValueError("Feature selector must be fitted before processing")

            if hasattr(self.selector, 'pca') and self.selector.pca is not None:
                n_features = self.selector.pca.n_components_
            else:
                n_features = len(self.selector.selected_indices)
            n_batches = (n_traces + batch_size - 1) // batch_size

            if output_file:
                with h5py.File(output_file, 'w') as out_f:
                    out_traces = out_f.create_dataset(
                        'traces',
                        shape=(n_traces, n_features),
                        dtype=np.float32,
                        chunks=(min(chunk_size, n_traces), n_features),
                        compression=compression if compression != 'None' else None
                    )

                    out_labels = out_f.create_dataset(
                        'labels',
                        shape=(n_traces,),
                        dtype=f['labels'].dtype,
                        chunks=(min(chunk_size * n_features // n_samples, n_traces),),
                        compression=compression if compression != 'None' else None
                    )

                    out_f.attrs['original_samples'] = n_samples
                    out_f.attrs['selected_features'] = n_features
                    out_f.attrs['preprocessing_config'] = str(self.preprocessor.__dict__)
                    if self.selector.selected_indices is not None:
                        out_f.attrs['selected_indices'] = self.selector.selected_indices
                    elif hasattr(self.selector, 'pca') and self.selector.pca is not None:
                        out_f.attrs['pca_components'] = self.selector.pca.n_components_

                    with tqdm(total=n_traces, desc="Processing traces") as pbar:
                        for i in range(n_batches):
                            start_idx = i * batch_size
                            end_idx = min((i + 1) * batch_size, n_traces)
                            batch_size_actual = end_idx - start_idx

                            batch_traces = f['traces'][start_idx:end_idx]
                            batch_labels = f['labels'][start_idx:end_idx]

                            try:
                                processed = self.transform(batch_traces)
                                out_traces[start_idx:end_idx] = processed
                                out_labels[start_idx:end_idx] = batch_labels
                            except Exception as e:
                                print(f"Error processing batch {i+1}/{n_batches}: {e}")
                                out_traces[start_idx:end_idx] = 0
                                out_labels[start_idx:end_idx] = batch_labels

                            pbar.update(batch_size_actual)

                    print(f"Processed data saved to {output_file}")

            else:
                with tqdm(total=n_traces, desc="Processing traces") as pbar:
                    for i in range(n_batches):
                        start_idx = i * batch_size
                        end_idx = min((i + 1) * batch_size, n_traces)

                        batch_traces = f['traces'][start_idx:end_idx]
                        _ = self.transform(batch_traces)

                        pbar.update(end_idx - start_idx)

        stats['processing_time'] = time.time() - start_time
        stats['traces_per_second'] = n_traces / stats['processing_time']

        return stats
