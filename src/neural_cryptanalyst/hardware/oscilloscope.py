import numpy as np

class OscilloscopeInterface:

    def capture_traces(self, num_traces: int, trigger_settings: dict):
        raise NotImplementedError("Subclass must implement")

class MockOscilloscope(OscilloscopeInterface):

    def capture_traces(self, num_traces: int, trigger_settings: dict):
        return np.random.randn(num_traces, 10000)
