import numpy as np
from neural_cryptanalyst.attacks.profiled import ProfiledAttack
from neural_cryptanalyst.models import SideChannelCNN
from neural_cryptanalyst import (
    TracePreprocessor,
    FeatureSelector,
    calculate_guessing_entropy as _calculate_guessing_entropy,
    calculate_success_rate as _calculate_success_rate,
)

def align_traces(traces, reference_trace=None):
    if traces.size == 0:
        raise IndexError("no traces provided")
    prep = TracePreprocessor()
    aligned, _ = prep.align_traces_correlation(traces, reference_trace)
    return aligned

def preprocess_traces(traces):
    prep = TracePreprocessor()
    prep.fit(traces)
    processed = prep.preprocess_traces(traces)
    return np.nan_to_num(processed)

def select_points_of_interest(traces, labels, num_poi=5):
    selector = FeatureSelector()
    return selector.select_poi_sost(traces, labels, num_poi=num_poi)

def calculate_guessing_entropy(preds, correct_key, num_traces_list):
    try:
        return _calculate_guessing_entropy(preds, correct_key, num_traces_list)
    except Exception:
        return np.ones(len(num_traces_list)) * 128

def calculate_success_rate(preds, correct_key, num_traces_list, rank_threshold=1):
    try:
        return _calculate_success_rate(preds, correct_key, num_traces_list, rank_threshold)
    except Exception:
        return np.zeros(len(num_traces_list))

def main() -> None:
    trace_length = 200
    num_traces = 50
    traces = np.random.randn(num_traces, trace_length).astype(np.float32)
    labels = np.random.randint(0, 256, num_traces)

    attack = ProfiledAttack(model=SideChannelCNN(trace_length=trace_length))
    attack.train_model(traces, labels, epochs=1, batch_size=10)
    preds = attack.attack(traces[:10])
    print("Predictions shape:", preds.shape)

if __name__ == "__main__":
    main()
