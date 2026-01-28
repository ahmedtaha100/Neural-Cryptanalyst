__version__ = "0.2.0"

from .preprocessing.trace_preprocessor import TracePreprocessor
from .preprocessing.feature_selector import FeatureSelector
from .preprocessing.augmentation import TraceAugmenter
from .preprocessing.config import PreprocessingConfig
from .preprocessing.pipeline import TracePipeline
from .models.cnn import SideChannelCNN
from .models.lstm import SideChannelLSTM
from .models.cnn_lstm import SideChannelCNNLSTM
from .models.transformer import SideChannelTransformer
from .models.gpam import GPAM, create_multi_task_labels
from .models.transnet import TransNet, create_shift_invariant_model, ShiftAugmentation
from .models.power_model import PowerConsumptionModel
from .models.losses import ranking_loss, focal_loss_ratio, cross_entropy_ratio
from .utils.crypto import aes_sbox, hamming_weight, hamming_distance
from .attacks.metrics import calculate_guessing_entropy, calculate_success_rate
from .attacks.profiled import ProfiledAttack
from .attacks.nonprofiled import NonProfiledAttack
from .attacks.ensemble import EnsembleAttack
from .datasets import ASCADDataset, DPAContestDataset, DatasetDownloader
from .detection.detector import SideChannelDetector
from .countermeasures import MaskingCountermeasure, HidingCountermeasure
from .hardware.oscilloscope import OscilloscopeInterface, MockOscilloscope
from .benchmarks.benchmark import AttackBenchmark
from .visualization.plots import (
    plot_power_traces, plot_correlation_results,
    plot_guessing_entropy_evolution, plot_snr_analysis,
    plot_attention_weights
)

def align_traces(traces, reference_trace=None):
    if getattr(traces, "size", 0) == 0:
        raise IndexError("no traces provided")
    prep = TracePreprocessor()
    aligned, _ = prep.align_traces_correlation(traces, reference_trace)
    return aligned

def preprocess_traces(traces):
    prep = TracePreprocessor()
    prep.fit(traces)
    return prep.preprocess_traces(traces)

def select_points_of_interest(traces, labels, num_poi=5):
    selector = FeatureSelector()
    idx, poi_traces = selector.select_poi_sost(traces, labels, num_poi=num_poi)
    return idx, poi_traces

__all__ = [
    'TracePreprocessor', 'FeatureSelector', 'TraceAugmenter', 'PreprocessingConfig', 'TracePipeline',
    'SideChannelCNN', 'SideChannelLSTM', 'SideChannelCNNLSTM', 'SideChannelTransformer',
    'GPAM', 'create_multi_task_labels', 'TransNet', 'create_shift_invariant_model', 'ShiftAugmentation',
    'PowerConsumptionModel', 'ranking_loss', 'focal_loss_ratio', 'cross_entropy_ratio',
    'calculate_guessing_entropy', 'calculate_success_rate', 'ProfiledAttack', 'NonProfiledAttack',
    'EnsembleAttack', 'ASCADDataset', 'DPAContestDataset', 'DatasetDownloader',
    'SideChannelDetector', 'MaskingCountermeasure', 'HidingCountermeasure', 'OscilloscopeInterface',
    'MockOscilloscope', 'AttackBenchmark', 'aes_sbox', 'hamming_weight', 'hamming_distance',
    'plot_power_traces', 'plot_correlation_results', 'plot_guessing_entropy_evolution', 'plot_snr_analysis',
    'plot_attention_weights', 'align_traces', 'preprocess_traces', 'select_points_of_interest'
]
