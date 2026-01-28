import numpy as np
import os
from neural_cryptanalyst import ProfiledAttack, SideChannelCNN
from neural_cryptanalyst.datasets import ASCADDataset

def main():
    ascad_path = 'ASCAD_data/ASCAD.h5'
    if not os.path.exists(ascad_path):
        print(f"Error: ASCAD dataset not found at {ascad_path}")
        print("Please download from: https://github.com/ANSSI-FR/ASCAD")
        print("And place in ASCAD_data/ directory")
        return

    dataset = ASCADDataset()
    train_traces, train_labels = dataset.load_ascad_v1(ascad_path)

    attack = ProfiledAttack(model=SideChannelCNN(trace_length=700))
    attack.train_model(train_traces[:45000], train_labels[:45000],
                       validation_split=0.1, num_features=700, epochs=50)

    attack.model.save_model('models/ascad_trained_cnn')

    attack_traces, attack_labels, attack_metadata = dataset.get_attack_set(ascad_path)

    predictions = attack.attack(attack_traces[:100])
    print(f"Attack completed. Predictions shape: {predictions.shape}")

if __name__ == "__main__":
    main()
