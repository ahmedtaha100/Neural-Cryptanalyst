import numpy as np
import time
import json
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import h5py

from ..models import (
    SideChannelCNN, SideChannelLSTM, SideChannelCNNLSTM,
    SideChannelTransformer, GPAM, TransNet
)
from ..attacks.profiled import ProfiledAttack
from ..preprocessing import TracePreprocessor, FeatureSelector
from ..datasets import ASCADDataset
from ..datasets.download import DatasetDownloader
from ..attacks.metrics import calculate_guessing_entropy, calculate_success_rate

class PaperBenchmarks:

    def __init__(self, results_dir: str = './benchmark_results'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.results = {}

    def benchmark_trace_reduction(self):
        print("\n=== Benchmarking Trace Reduction ===")
        from ..attacks.nonprofiled import NonProfiledAttack
        n_traces = 20000
        trace_length = 5000
        traces = np.random.randn(n_traces, trace_length) * 0.1
        keys = np.random.randint(0, 256, n_traces)
        plaintexts = np.random.randint(0, 256, n_traces)
        for i in range(n_traces):
            from ..utils.crypto import aes_sbox
            hw = bin(aes_sbox(plaintexts[i] ^ keys[i])).count('1')
            traces[i, 2000:2100] += hw * 0.3
        print("Running traditional CPA...")
        nonprofiled = NonProfiledAttack()
        cpa_traces_needed = 0
        for n in [100, 500, 1000, 2000, 5000, 10000]:
            correlations = nonprofiled.dpa_attack(traces[:n], plaintexts[:n])
            if np.argmax(correlations) == keys[0]:
                cpa_traces_needed = n
                break
        print(f"Traditional CPA needed: {cpa_traces_needed} traces")
        print("Running ML-based attack...")
        train_traces = traces[:15000]
        train_labels = keys[:15000]
        test_traces = traces[15000:]
        test_labels = keys[15000:]
        attack = ProfiledAttack(model=SideChannelCNN(trace_length=trace_length))
        attack.model.compile_model()
        train_reshaped = train_traces.reshape(-1, trace_length, 1)
        train_labels_cat = tf.keras.utils.to_categorical(train_labels, 256)
        attack.model.model.fit(
            train_reshaped[:5000], train_labels_cat[:5000],
            validation_split=0.2,
            epochs=20,
            batch_size=64,
            verbose=0
        )
        ml_traces_needed = 0
        for n in [10, 50, 100, 200, 500, 1000]:
            test_batch = test_traces[:n].reshape(-1, trace_length, 1)
            predictions = attack.model.model.predict(test_batch, verbose=0)
            accumulated = np.mean(predictions, axis=0)
            if np.argmax(accumulated) == test_labels[0]:
                ml_traces_needed = n
                break
        print(f"ML-based attack needed: {ml_traces_needed} traces")
        if cpa_traces_needed > 0:
            reduction = (1 - ml_traces_needed / cpa_traces_needed) * 100
            print(f"Reduction: {reduction:.1f}%")
        else:
            reduction = 0
        self.results['trace_reduction'] = {
            'traditional_cpa_traces': cpa_traces_needed,
            'ml_traces': ml_traces_needed,
            'reduction_percentage': reduction
        }
        return reduction

    def benchmark_architectures_comparison(self):
        print("\n=== Benchmarking Architecture Comparison ===")
        n_traces = 10000
        trace_length = 2000
        traces = np.random.randn(n_traces, trace_length) * 0.1
        keys = np.random.randint(0, 256, n_traces)
        for i in range(n_traces):
            hw = bin(keys[i]).count('1')
            traces[i, 500:600] += hw * 0.4
            traces[i, 1000:1100] += hw * 0.3
        train_traces = traces[:8000].reshape(-1, trace_length, 1)
        train_labels = tf.keras.utils.to_categorical(keys[:8000], 256)
        test_traces = traces[8000:].reshape(-1, trace_length, 1)
        test_labels = keys[8000:]
        architectures = {
            'CNN': SideChannelCNN(trace_length=trace_length),
            'LSTM': SideChannelLSTM(trace_length=trace_length),
            'CNN-LSTM': SideChannelCNNLSTM(trace_length=trace_length),
            'Transformer': SideChannelTransformer(trace_length=trace_length, d_model=128, num_heads=4)
        }
        results = {}
        for name, model in architectures.items():
            print(f"\nTraining {name}...")
            model.compile_model()
            start_time = time.time()
            if hasattr(model, 'train'):
                history = model.train(
                    train_traces, train_labels,
                    train_traces[:1000], train_labels[:1000],
                    epochs=10,
                    batch_size=64
                )
            else:
                history = model.model.fit(
                    train_traces, train_labels,
                    validation_split=0.2,
                    epochs=10,
                    batch_size=64,
                    verbose=0
                )
            train_time = time.time() - start_time
            trace_counts = [100, 200, 500, 1000]
            success_rates = []
            for n_traces in trace_counts:
                predictions = model.model.predict(test_traces[:n_traces], verbose=0)
                success = 0
                for i in range(min(10, len(test_labels) // n_traces)):
                    batch_preds = predictions[i*10:(i+1)*10] if n_traces >= 10 else predictions
                    accumulated = np.mean(batch_preds, axis=0)
                    if np.argmax(accumulated) == test_labels[i*10]:
                        success += 1
                success_rate = (success / min(10, len(test_labels) // n_traces)) * 100
                success_rates.append(success_rate)
            results[name] = {
                'train_time': train_time,
                'model_parameters': model.model.count_params(),
                'trace_counts': trace_counts,
                'success_rates': success_rates,
                'final_accuracy': float(history.history['accuracy'][-1])
            }
            print(f"{name} - Parameters: {results[name]['model_parameters']:,}")
            print(f"{name} - Training time: {train_time:.2f}s")
            print(f"{name} - Success rates: {success_rates}")
        plt.figure(figsize=(12, 8))
        for name, data in results.items():
            plt.plot(data['trace_counts'], data['success_rates'],
                     marker='o', linewidth=2, markersize=8, label=name)
        plt.xlabel('Number of Traces', fontsize=12)
        plt.ylabel('Success Rate (%)', fontsize=12)
        plt.title('Architecture Comparison - Success Rate vs Trace Count', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'architecture_comparison.png'), dpi=300)
        self.results['architecture_comparison'] = results
        return results

    def benchmark_masked_implementations(self):
        print("\n=== Benchmarking Masked Implementation Attacks ===")
        downloader = DatasetDownloader('./data')
        results = {}
        for masking_order in [0, 1, 2]:
            print(f"\nTesting {masking_order}-order masked implementation...")
            dataset_path = downloader.generate_synthetic_masked_aes(
                num_traces=20000,
                trace_length=3000,
                masking_order=masking_order
            )
            with h5py.File(dataset_path, 'r') as f:
                traces = f['traces'][:]
                keys = f['keys'][:]
            preprocessor = TracePreprocessor()
            selector = FeatureSelector()
            train_traces = traces[:15000]
            train_labels = keys[:15000]
            test_traces = traces[15000:]
            test_labels = keys[15000:]
            preprocessor.fit(train_traces)
            train_processed = preprocessor.preprocess_traces(train_traces)
            test_processed = preprocessor.preprocess_traces(test_traces)
            _, train_selected = selector.select_poi_sost(train_processed, train_labels, num_poi=1000)
            test_selected = selector.transform(test_processed)
            attack = ProfiledAttack(model=SideChannelCNN(trace_length=1000))
            attack.train_model(train_selected, train_labels, epochs=30, batch_size=64)
            if masking_order == 0:
                trace_counts = [50, 100, 200]
            elif masking_order == 1:
                trace_counts = [500, 1000, 1500]
            else:
                trace_counts = [2000, 2500, 3000]
            success_rates = []
            for n_traces in trace_counts:
                predictions = attack.attack(test_selected[:n_traces])
                accumulated = np.mean(predictions, axis=0)
                success = np.argmax(accumulated) == test_labels[0]
                success_rates.append(int(success))
            results[f'order_{masking_order}'] = {
                'trace_counts': trace_counts,
                'success_rates': success_rates,
                'traces_to_break': trace_counts[success_rates.index(1)] if 1 in success_rates else None
            }
            print(f"Order-{masking_order} results: {results[f'order_{masking_order}']}")
        self.results['masked_implementations'] = results
        return results

    def run_all_benchmarks(self):
        print("="*60)
        print("Running Complete Benchmark Suite")
        print("="*60)
        self.benchmark_trace_reduction()
        self.benchmark_architectures_comparison()
        self.benchmark_masked_implementations()
        results_file = os.path.join(self.results_dir, 'benchmark_results.json')
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nAll results saved to: {results_file}")
        self.generate_report()

    def generate_report(self):
        report_path = os.path.join(self.results_dir, 'benchmark_report.txt')
        with open(report_path, 'w') as f:
            f.write("Neural Cryptanalyst Benchmark Report\n")
            f.write("="*50 + "\n\n")
            if 'trace_reduction' in self.results:
                f.write("1. Trace Reduction Benchmark\n")
                f.write("-"*30 + "\n")
                data = self.results['trace_reduction']
                f.write(f"Traditional CPA: {data['traditional_cpa_traces']} traces\n")
                f.write(f"ML-based: {data['ml_traces']} traces\n")
                f.write(f"Reduction: {data['reduction_percentage']:.1f}%\n\n")
            if 'architecture_comparison' in self.results:
                f.write("2. Architecture Comparison\n")
                f.write("-"*30 + "\n")
                for arch, data in self.results['architecture_comparison'].items():
                    f.write(f"\n{arch}:\n")
                    f.write(f"  Parameters: {data['model_parameters']:,}\n")
                    f.write(f"  Training time: {data['train_time']:.2f}s\n")
                    f.write(f"  Success rates: {data['success_rates']}\n")
            if 'masked_implementations' in self.results:
                f.write("\n3. Masked Implementation Attacks\n")
                f.write("-"*30 + "\n")
                for order, data in self.results['masked_implementations'].items():
                    f.write(f"\n{order}:\n")
                    f.write(f"  Traces to break: {data['traces_to_break']}\n")
        print(f"Report saved to: {report_path}")

if __name__ == "__main__":
    benchmarks = PaperBenchmarks()
    benchmarks.run_all_benchmarks()
