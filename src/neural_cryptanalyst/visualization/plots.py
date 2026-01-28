import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Tuple

def plot_power_traces(traces: np.ndarray, labels: Optional[np.ndarray] = None, title: str = "Power Traces", max_traces: int = 10):
    fig, ax = plt.subplots(figsize=(12, 6))
    n_traces = min(len(traces), max_traces)
    for i in range(n_traces):
        label = f"Trace {i}" if labels is None else f"Trace {i} (Key: {labels[i]})"
        ax.plot(traces[i], alpha=0.7, label=label)
    ax.set_xlabel("Sample Points")
    ax.set_ylabel("Power Consumption")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

def plot_correlation_results(correlations: np.ndarray, correct_key: int, title: str = "CPA Results"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    for key in range(256):
        if key == correct_key:
            ax1.plot(correlations[key], 'r-', linewidth=2, label=f'Correct Key ({key})')
        else:
            ax1.plot(correlations[key], 'b-', alpha=0.1, linewidth=0.5)
    ax1.set_xlabel("Sample Points")
    ax1.set_ylabel("Correlation")
    ax1.set_title(f"{title} - All Key Hypotheses")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    max_correlations = np.max(np.abs(correlations), axis=1)
    ax2.bar(range(256), max_correlations, color=['red' if k == correct_key else 'blue' for k in range(256)])
    ax2.set_xlabel("Key Hypothesis")
    ax2.set_ylabel("Maximum Correlation")
    ax2.set_title("Maximum Correlation per Key")
    ax2.grid(True, alpha=0.3)
    return fig

def plot_guessing_entropy_evolution(guessing_entropy: np.ndarray, num_traces: List[int], title: str = "Guessing Entropy Evolution"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(num_traces, guessing_entropy, 'b-', marker='o', markersize=8, linewidth=2)
    ax.set_xlabel("Number of Traces")
    ax.set_ylabel("Guessing Entropy (bits)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.axhline(y=0, color='g', linestyle='--', alpha=0.5, label='Key Found')
    ax.axhline(y=np.log2(256), color='r', linestyle='--', alpha=0.5, label='Random Guess')
    ax.legend()
    return fig

def plot_snr_analysis(traces: np.ndarray, labels: np.ndarray, title: str = "Signal-to-Noise Ratio"):
    unique_labels = np.unique(labels)
    class_means = np.array([np.mean(traces[labels == label], axis=0) for label in unique_labels])
    signal_var = np.var(class_means, axis=0)
    noise_var = np.mean([np.var(traces[labels == label], axis=0) for label in unique_labels], axis=0)
    snr = signal_var / (noise_var + 1e-10)
    snr_db = 10 * np.log10(snr + 1e-10)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax1.plot(traces[0], 'b-', alpha=0.7)
    ax1.set_ylabel("Power Consumption")
    ax1.set_title(f"{title} - Example Trace")
    ax1.grid(True, alpha=0.3)
    ax2.plot(snr_db, 'r-', linewidth=2)
    ax2.set_xlabel("Sample Points")
    ax2.set_ylabel("SNR (dB)")
    ax2.set_title("SNR Analysis")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    return fig

def plot_attention_weights(attention_weights: np.ndarray, trace_length: int, title: str = "Transformer Attention Weights"):
    fig, ax = plt.subplots(figsize=(12, 8))
    avg_attention = np.mean(attention_weights, axis=0)
    im = ax.imshow(avg_attention, cmap='hot', aspect='auto')
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Attention Weight")
    return fig
