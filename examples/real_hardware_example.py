import numpy as np

from neural_cryptanalyst import ProfiledAttack, SideChannelCNN

def capture_from_chipwhisperer():

    print("This would interface with ChipWhisperer")
    return np.random.randn(1000, 5000), np.random.randint(0, 256, 1000)

def main():
    print("=== Real Hardware Attack Example ===")
    traces, keys = capture_from_chipwhisperer()

    attack = ProfiledAttack(model=SideChannelCNN(trace_length=5000))
    attack.train_model(traces, keys, epochs=1, batch_size=32)

    preds = attack.attack(traces[:10])
    print(f"Predictions shape: {preds.shape}")

if __name__ == "__main__":
    main()
