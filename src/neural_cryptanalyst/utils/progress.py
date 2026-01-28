from tqdm import tqdm

class AttackProgress:
    def __init__(self, total_traces: int):
        self.pbar = tqdm(total=total_traces, desc="Attack Progress")

    def update(self, n: int = 1):
        self.pbar.update(n)

    def close(self):
        self.pbar.close()
