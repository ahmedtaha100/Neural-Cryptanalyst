from sklearn.decomposition import PCA, IncrementalPCA
from typing import Union, Optional, List
import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from typing import Tuple
import warnings

class FeatureSelector:

    def __init__(self, n_jobs: int = 1):
        self.n_jobs = n_jobs
        self.selected_indices = None
        self.selection_scores = None

    def select_poi_sost(self, traces: np.ndarray, labels: np.ndarray,
                        num_poi: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        n_traces, n_samples = traces.shape
        unique_labels = np.unique(labels)

        class_means = {}
        class_counts = {}

        for label in unique_labels:
            mask = labels == label
            class_counts[label] = np.sum(mask)
            class_means[label] = np.mean(traces[mask], axis=0)

        global_mean = np.mean(traces, axis=0)

        sost = np.zeros(n_samples)
        for label in unique_labels:
            sost += class_counts[label] * np.square(class_means[label] - global_mean)

        self.selection_scores = sost
        self.selected_indices = np.argsort(sost)[-num_poi:]

        return self.selected_indices, traces[:, self.selected_indices]

    def select_poi_ttest(self, traces: np.ndarray, labels: np.ndarray,
                         num_poi: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        unique_labels = np.unique(labels)
        if len(unique_labels) != 2:
            raise ValueError("T-test requires exactly 2 classes")

        mask0 = labels == unique_labels[0]
        mask1 = labels == unique_labels[1]

        t_stats, _ = stats.ttest_ind(traces[mask0], traces[mask1],
                                     equal_var=False, axis=0)
        t_scores = np.abs(t_stats)

        self.selection_scores = t_scores
        self.selected_indices = np.argsort(t_scores)[-num_poi:]

        return self.selected_indices, traces[:, self.selected_indices]

    def select_poi_mutual_information(self, traces: np.ndarray, labels: np.ndarray,
                                      num_poi: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        mi_scores = np.zeros(traces.shape[1])

        for i in range(traces.shape[1]):
            mi_scores[i] = mutual_info_classif(traces[:, i].reshape(-1, 1),
                                              labels, random_state=42)[0]

        self.selection_scores = mi_scores
        self.selected_indices = np.argsort(mi_scores)[-num_poi:]

        return self.selected_indices, traces[:, self.selected_indices]

    def transform(self, traces: np.ndarray) -> np.ndarray:
        if hasattr(self, 'pca') and self.pca is not None:
            return self.pca.transform(traces)
        elif self.selected_indices is not None:
            return traces[:, self.selected_indices]
        else:
            raise ValueError("No feature selection or PCA has been fitted")

    def apply_pca(self, traces: np.ndarray, n_components: Union[int, float] = 0.95,
                  batch_size: Optional[int] = None) -> np.ndarray:
        if isinstance(n_components, float) and not 0 < n_components <= 1:
            raise ValueError("When float, n_components must be in (0, 1]")

        if batch_size is not None:
            if isinstance(n_components, float):
                sample_size = min(batch_size * 2, len(traces))
                sample_pca = PCA(n_components=n_components)
                sample_pca.fit(traces[:sample_size])
                n_components = sample_pca.n_components_
                warnings.warn(
                    f"IncrementalPCA requires integer n_components. "
                    f"Estimated {n_components} components from variance ratio."
                )

            n_components = min(n_components, traces.shape[1])
            self.pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
            n_batches = (len(traces) + batch_size - 1) // batch_size
            for i in range(n_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, len(traces))
                self.pca.partial_fit(traces[start:end])
            transformed = self.pca.transform(traces)
        else:
            if isinstance(n_components, int):
                n_components = min(n_components, traces.shape[0], traces.shape[1])
            self.pca = PCA(n_components=n_components)
            transformed = self.pca.fit_transform(traces)

        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        self.selected_indices = None
        return transformed

    def select_poi_cpa(self, traces: np.ndarray, labels: np.ndarray, num_poi: int = 1000, leakage_model: str = 'hw') -> Tuple[np.ndarray, np.ndarray]:
        if leakage_model == 'hw':
            hypotheses = np.array([bin(label).count('1') for label in labels])
        else:
            raise NotImplementedError("Only HW model implemented for POI selection")
        correlations = np.zeros(traces.shape[1])
        for i in range(traces.shape[1]):
            correlations[i] = np.abs(stats.pearsonr(traces[:, i], hypotheses)[0])
        self.selection_scores = correlations
        self.selected_indices = np.argsort(correlations)[-num_poi:]
        return self.selected_indices, traces[:, self.selected_indices]

    def select_poi_snr(self, traces: np.ndarray, labels: np.ndarray, num_poi: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        unique_labels = np.unique(labels)
        signal_var = np.zeros(traces.shape[1])
        noise_var = np.zeros(traces.shape[1])
        class_means = np.zeros((len(unique_labels), traces.shape[1]))
        for i, label in enumerate(unique_labels):
            mask = labels == label
            class_means[i] = np.mean(traces[mask], axis=0)
            noise_var += np.sum(np.var(traces[mask], axis=0))
        signal_var = np.var(class_means, axis=0)
        noise_var /= len(unique_labels)
        snr = signal_var / (noise_var + 1e-10)
        self.selection_scores = snr
        self.selected_indices = np.argsort(snr)[-num_poi:]
        return self.selected_indices, traces[:, self.selected_indices]

    def combine_methods(self, traces: np.ndarray, labels: np.ndarray, methods: List[str] = ['sost', 'ttest', 'mi'], num_poi: int = 1000, combination: str = 'union') -> Tuple[np.ndarray, np.ndarray]:
        from functools import reduce
        all_indices = []
        all_scores = {}
        for method in methods:
            if method == 'sost':
                indices, _ = self.select_poi_sost(traces, labels, num_poi)
            elif method == 'ttest':
                indices, _ = self.select_poi_ttest(traces, labels, num_poi)
            elif method == 'mi':
                indices, _ = self.select_poi_mutual_information(traces, labels, num_poi)
            elif method == 'cpa':
                indices, _ = self.select_poi_cpa(traces, labels, num_poi)
            elif method == 'snr':
                indices, _ = self.select_poi_snr(traces, labels, num_poi)
            else:
                warnings.warn(f"Unknown method {method}, skipping")
                continue
            all_indices.append(indices)
            all_scores[method] = self.selection_scores
        if combination == 'union':
            combined_indices = np.unique(np.concatenate(all_indices))[:num_poi]
        elif combination == 'intersection':
            combined_indices = reduce(np.intersect1d, all_indices)
            if len(combined_indices) < num_poi:
                warnings.warn(f"Intersection has only {len(combined_indices)} POIs")
        elif combination == 'rank_fusion':
            rank_sum = np.zeros(traces.shape[1])
            for method, scores in all_scores.items():
                ranks = stats.rankdata(scores)
                rank_sum += ranks
            combined_indices = np.argsort(rank_sum)[-num_poi:]
        else:
            raise ValueError(f"Unknown combination method: {combination}")
        self.selected_indices = combined_indices
        return combined_indices, traces[:, combined_indices]

    def cross_validate_poi_selection(self, traces: np.ndarray, labels: np.ndarray,
                                     num_poi_list: List[int], cv: int = 5) -> dict:
        from sklearn.model_selection import KFold
        from sklearn.svm import SVC

        results = {}
        kfold = KFold(n_splits=cv, shuffle=True, random_state=42)

        for num_poi in num_poi_list:
            scores = []

            for train_idx, val_idx in kfold.split(traces):
                train_traces = traces[train_idx]
                train_labels = labels[train_idx]
                val_traces = traces[val_idx]
                val_labels = labels[val_idx]

                _, train_selected = self.select_poi_sost(train_traces, train_labels, num_poi)

                val_selected = val_traces[:, self.selected_indices]

                clf = SVC(kernel='linear', C=1.0, random_state=42)
                clf.fit(train_selected, train_labels)

                score = clf.score(val_selected, val_labels)
                scores.append(score)

            results[num_poi] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores
            }

        return results
