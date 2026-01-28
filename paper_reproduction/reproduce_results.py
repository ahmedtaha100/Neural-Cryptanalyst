import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_cryptanalyst import (
    ProfiledAttack, SideChannelCNN, SideChannelLSTM,
    SideChannelTransformer, TracePreprocessor, FeatureSelector,
    TraceAugmenter, calculate_guessing_entropy, calculate_success_rate
)
from neural_cryptanalyst.datasets import ASCADDataset
from neural_cryptanalyst.utils.crypto import aes_sbox
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

class PaperResultsReproduction:

    def __init__(self, ascad_path='./ASCAD_data/ASCAD.h5'):
        self.ascad_path = ascad_path
        self.results = {}

    def reproduce_cnn_masked_aes_attack(self):

        print("\n=== Reproducing CNN Attack on Masked AES ===")
        print("Target: 92% success rate with 500 traces")

        dataset = ASCADDataset()
        traces, labels = dataset.load_ascad_v1(self.ascad_path)

        preprocessor = TracePreprocessor()
        selector = FeatureSelector()

        train_traces = traces[:45000]
        train_labels = labels[:45000]
        test_traces = traces[45000:50000]
        test_labels = labels[45000:50000]

        preprocessor.fit(train_traces)
        processed_train = preprocessor.preprocess_traces(train_traces)
        processed_test = preprocessor.preprocess_traces(test_traces)

        _, selected_train = selector.select_poi_sost(processed_train, train_labels, num_poi=700)
        selected_test = selector.transform(processed_test)

        attack = ProfiledAttack(model=SideChannelCNN(trace_length=700))
        attack.train_model(selected_train, train_labels, epochs=75, batch_size=100)

        success_rates = []
        trace_counts = [100, 200, 300, 400, 500, 750, 1000]

        for n_traces in trace_counts:
            sr_list = []
            for exp in range(100):
                start_idx = exp * 10
                attack_traces = selected_test[start_idx:start_idx + n_traces]
                predictions = attack.attack(attack_traces, num_attack_traces=n_traces)

                predicted_key = np.argmax(np.mean(predictions, axis=0))
                correct = predicted_key == test_labels[start_idx]
                sr_list.append(correct)

            success_rate = np.mean(sr_list) * 100
            success_rates.append(success_rate)
            print(f"Traces: {n_traces}, Success Rate: {success_rate:.1f}%")

        plt.figure(figsize=(10, 6))
        plt.plot(trace_counts, success_rates, 'b-o', linewidth=2, markersize=8)
        plt.axhline(y=92, color='r', linestyle='--', label='Target: 92%')
        plt.xlabel('Number of Traces')
        plt.ylabel('Success Rate (%)')
        plt.title('CNN Attack on First-Order Masked AES (ASCAD)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig('results/cnn_masked_aes_success_rate.png')

        self.results['cnn_masked_aes'] = {
            'trace_counts': trace_counts,
            'success_rates': success_rates,
            'achieved_500_traces': success_rates[trace_counts.index(500)]
        }

        return success_rates

    def reproduce_transformer_attack(self):

        print("\n=== Reproducing Transformer Attack ===")
        print("Target: 96% success rate with 300 traces")

        n_traces = 50000
        trace_length = 5000

        traces = np.random.randn(n_traces, trace_length) * 0.1
        keys = np.random.randint(0, 256, n_traces)

        for i in range(n_traces):
            hw = bin(aes_sbox(keys[i])).count('1')
            leakage_points = [1000, 1500, 2000, 2500]
            for point in leakage_points:
                traces[i, point:point+50] += hw * 0.5

        train_traces = traces[:40000]
        train_labels = keys[:40000]
        test_traces = traces[40000:]
        test_labels = keys[40000:]

        preprocessor = TracePreprocessor()
        selector = FeatureSelector()

        preprocessor.fit(train_traces)
        processed_train = preprocessor.preprocess_traces(train_traces)
        processed_test = preprocessor.preprocess_traces(test_traces)

        _, selected_train = selector.select_poi_sost(processed_train, train_labels, num_poi=1000)
        selected_test = selector.transform(processed_test)

        selected_train = selected_train.reshape(selected_train.shape[0], selected_train.shape[1], 1)
        selected_test = selected_test.reshape(selected_test.shape[0], selected_test.shape[1], 1)

        transformer = SideChannelTransformer.create_gpam_variant(trace_length=1000)
        transformer.compile_model(learning_rate=0.0001)

        train_labels_cat = tf.keras.utils.to_categorical(train_labels, num_classes=256)

        history = transformer.model.fit(
            selected_train, train_labels_cat,
            validation_split=0.1,
            epochs=50,
            batch_size=64,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            ],
            verbose=1
        )

        success_rates = []
        trace_counts = [50, 100, 200, 300, 400, 500]

        for n_traces in trace_counts:
            sr_list = []
            for exp in range(100):
                start_idx = exp * 5
                attack_traces = selected_test[start_idx:start_idx + n_traces]
                predictions = transformer.model.predict(attack_traces, verbose=0)

                accumulated = np.mean(predictions, axis=0)
                predicted_key = np.argmax(accumulated)
                correct = predicted_key == test_labels[start_idx]
                sr_list.append(correct)

            success_rate = np.mean(sr_list) * 100
            success_rates.append(success_rate)
            print(f"Traces: {n_traces}, Success Rate: {success_rate:.1f}%")

        self.results['transformer'] = {
            'trace_counts': trace_counts,
            'success_rates': success_rates,
            'achieved_300_traces': success_rates[trace_counts.index(300)] if 300 in trace_counts else 'N/A'
        }

        return success_rates

    def reproduce_lstm_misalignment_attack(self):

        print("\n=== Reproducing LSTM Attack on Misaligned Traces ===")
        print("Target: 85% success rate with 1000-sample misalignment")

        n_traces = 20000
        trace_length = 10000

        traces = np.random.randn(n_traces, trace_length) * 0.1
        keys = np.random.randint(0, 256, n_traces)

        for i in range(n_traces):
            hw = bin(aes_sbox(keys[i])).count('1')
            misalignment = np.random.randint(-500, 500)
            leakage_point = 5000 + misalignment

            if 0 < leakage_point < trace_length - 100:
                traces[i, leakage_point:leakage_point+100] += hw * 0.3

        train_traces = traces[:15000]
        train_labels = keys[:15000]
        test_traces = traces[15000:]
        test_labels = keys[15000:]

        lstm_attack = ProfiledAttack(
            model=SideChannelLSTM(trace_length=trace_length, bidirectional=True)
        )

        lstm_attack.model.compile_model()

        train_traces = train_traces.reshape(train_traces.shape[0], train_traces.shape[1], 1)
        test_traces = test_traces.reshape(test_traces.shape[0], test_traces.shape[1], 1)
        train_labels_cat = tf.keras.utils.to_categorical(train_labels, num_classes=256)

        history = lstm_attack.model.train(
            train_traces, train_labels_cat,
            train_traces[:1000], train_labels_cat[:1000],
            epochs=30,
            batch_size=32
        )

        success_count = 0
        n_tests = 100

        for i in range(n_tests):
            test_batch = test_traces[i*10:(i+1)*10]
            predictions = lstm_attack.model.model.predict(test_batch, verbose=0)

            accumulated = np.mean(predictions, axis=0)
            predicted_key = np.argmax(accumulated)

            if predicted_key == test_labels[i*10]:
                success_count += 1

        success_rate = (success_count / n_tests) * 100
        print(f"Success rate with misaligned traces: {success_rate:.1f}%")

        self.results['lstm_misalignment'] = {
            'success_rate': success_rate,
            'misalignment_range': 1000
        }

        return success_rate

    def reproduce_ensemble_second_order_attack(self):

        print("\n=== Reproducing Ensemble Attack on Second-Order Masking ===")
        print("Target: Successful key recovery with 2000-3000 traces")

        from neural_cryptanalyst.attacks.ensemble import EnsembleAttack
        from neural_cryptanalyst.models import SideChannelCNNLSTM

        n_traces = 10000
        trace_length = 5000

        traces = np.random.randn(n_traces, trace_length) * 0.05
        keys = np.random.randint(0, 256, n_traces)

        for i in range(n_traces):
            mask1 = np.random.randint(0, 256)
            mask2 = np.random.randint(0, 256)

            masked_value1 = keys[i] ^ mask1
            masked_value2 = mask1 ^ mask2

            hw1 = bin(masked_value1).count('1')
            hw2 = bin(masked_value2).count('1')

            traces[i, 1000:1050] += hw1 * 0.1
            traces[i, 3000:3050] += hw2 * 0.1

            traces[i, 2000:2050] += (hw1 * hw2) * 0.05

        train_traces = traces[:7000]
        train_labels = keys[:7000]
        test_traces = traces[7000:]
        test_labels = keys[7000:]

        from neural_cryptanalyst.attacks.nonprofiled import NonProfiledAttack
        nonprofiled = NonProfiledAttack()

        train_2nd = np.zeros((len(train_traces), trace_length - 100))
        test_2nd = np.zeros((len(test_traces), trace_length - 100))

        for i in range(len(train_traces)):
            centered = train_traces[i] - np.mean(train_traces[i])
            train_2nd[i] = centered[:-100] * centered[100:]

        for i in range(len(test_traces)):
            centered = test_traces[i] - np.mean(test_traces[i])
            test_2nd[i] = centered[:-100] * centered[100:]

        models = [
            SideChannelCNN(trace_length=trace_length-100),
            SideChannelLSTM(trace_length=trace_length-100),
            SideChannelCNNLSTM(trace_length=trace_length-100)
        ]

        ensemble = EnsembleAttack(models=models)

        train_2nd = train_2nd.reshape(train_2nd.shape[0], train_2nd.shape[1], 1)
        test_2nd = test_2nd.reshape(test_2nd.shape[0], test_2nd.shape[1], 1)
        train_labels_cat = tf.keras.utils.to_categorical(train_labels, num_classes=256)

        ensemble.train_ensemble(
            train_2nd, train_labels_cat,
            epochs=20, batch_size=64,
            X_val=train_2nd[:500], y_val=train_labels_cat[:500]
        )

        trace_counts = [1000, 1500, 2000, 2500, 3000]
        success_rates = []

        for n_traces in trace_counts:
            attack_traces = test_2nd[:n_traces]
            predictions = ensemble.predict_ensemble(attack_traces, method='average')

            accumulated = np.mean(predictions, axis=0)
            predicted_key = np.argmax(accumulated)

            top_5 = np.argsort(accumulated)[-5:]
            success = test_labels[0] in top_5

            print(f"Traces: {n_traces}, Key in top-5: {success}")
            success_rates.append(success)

        self.results['second_order_masking'] = {
            'trace_counts': trace_counts,
            'success_rates': success_rates
        }

        return success_rates

def main():

    reproducer = PaperResultsReproduction()

    os.makedirs('results', exist_ok=True)

    if os.path.exists(reproducer.ascad_path):
        reproducer.reproduce_cnn_masked_aes_attack()
    else:
        print("ASCAD dataset not found. Please download it first.")

    reproducer.reproduce_transformer_attack()
    reproducer.reproduce_lstm_misalignment_attack()
    reproducer.reproduce_ensemble_second_order_attack()

    import json
    with open('results/reproduction_results.json', 'w') as f:
        json.dump(reproducer.results, f, indent=2)

    print("\n=== Reproduction Complete ===")
    print("Results saved to results/reproduction_results.json")

if __name__ == "__main__":
    main()
