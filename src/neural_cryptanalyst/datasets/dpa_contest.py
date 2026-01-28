class DPAContestDataset:
    def __init__(self):
        self.traces = None
        self.labels = None

    def load(self, traces, labels):
        self.traces = traces
        self.labels = labels
        return self.traces, self.labels
