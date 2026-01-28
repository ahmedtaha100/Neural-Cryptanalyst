import numpy as np
from typing import Dict, List, Tuple, Optional

class CompletePowerConsumptionModel:

    def __init__(self,
                 alpha: float = 0.5,
                 capacitance: float = 1e-12,
                 voltage: float = 3.3,
                 frequency: float = 100e6,
                 leakage_current: float = 1e-6,
                 short_circuit_current: float = 1e-7,
                 thermal_noise_density: float = 4e-21,
                 flicker_noise_coefficient: float = 1e-12,
                 num_bits: int = 8,
                 technology_node: float = 45e-9):
        self.alpha = alpha
        self.capacitance = capacitance
        self.voltage = voltage
        self.frequency = frequency
        self.leakage_current = leakage_current
        self.short_circuit_current = short_circuit_current
        self.thermal_noise_density = thermal_noise_density
        self.flicker_noise_coefficient = flicker_noise_coefficient
        self.num_bits = num_bits
        self.technology_node = technology_node
        self.temperature = 300
        self.k_boltzmann = 1.38e-23
        self.sbox = [
            0x63,0x7C,0x77,0x7B,0xF2,0x6B,0x6F,0xC5,0x30,0x01,0x67,0x2B,0xFE,0xD7,0xAB,0x76,
            0xCA,0x82,0xC9,0x7D,0xFA,0x59,0x47,0xF0,0xAD,0xD4,0xA2,0xAF,0x9C,0xA4,0x72,0xC0,
            0xB7,0xFD,0x93,0x26,0x36,0x3F,0xF7,0xCC,0x34,0xA5,0xE5,0xF1,0x71,0xD8,0x31,0x15,
            0x04,0xC7,0x23,0xC3,0x18,0x96,0x05,0x9A,0x07,0x12,0x80,0xE2,0xEB,0x27,0xB2,0x75,
            0x09,0x83,0x2C,0x1A,0x1B,0x6E,0x5A,0xA0,0x52,0x3B,0xD6,0xB3,0x29,0xE3,0x2F,0x84,
            0x53,0xD1,0x00,0xED,0x20,0xFC,0xB1,0x5B,0x6A,0xCB,0xBE,0x39,0x4A,0x4C,0x58,0xCF,
            0xD0,0xEF,0xAA,0xFB,0x43,0x4D,0x33,0x85,0x45,0xF9,0x02,0x7F,0x50,0x3C,0x9F,0xA8,
            0x51,0xA3,0x40,0x8F,0x92,0x9D,0x38,0xF5,0xBC,0xB6,0xDA,0x21,0x10,0xFF,0xF3,0xD2,
            0xCD,0x0C,0x13,0xEC,0x5F,0x97,0x44,0x17,0xC4,0xA7,0x7E,0x3D,0x64,0x5D,0x19,0x73,
            0x60,0x81,0x4F,0xDC,0x22,0x2A,0x90,0x88,0x46,0xEE,0xB8,0x14,0xDE,0x5E,0x0B,0xDB,
            0xE0,0x32,0x3A,0x0A,0x49,0x06,0x24,0x5C,0xC2,0xD3,0xAC,0x62,0x91,0x95,0xE4,0x79,
            0xE7,0xC8,0x37,0x6D,0x8D,0xD5,0x4E,0xA9,0x6C,0x56,0xF4,0xEA,0x65,0x7A,0xAE,0x08,
            0xBA,0x78,0x25,0x2E,0x1C,0xA6,0xB4,0xC6,0xE8,0xDD,0x74,0x1F,0x4B,0xBD,0x8B,0x8A,
            0x70,0x3E,0xB5,0x66,0x48,0x03,0xF6,0x0E,0x61,0x35,0x57,0xB9,0x86,0xC1,0x1D,0x9E,
            0xE1,0xF8,0x98,0x11,0x69,0xD9,0x8E,0x94,0x9B,0x1E,0x87,0xE9,0xCE,0x55,0x28,0xDF,
            0x8C,0xA1,0x89,0x0D,0xBF,0xE6,0x42,0x68,0x41,0x99,0x2D,0x0F,0xB0,0x54,0xBB,0x16
        ]

    def calculate_dynamic_power(self, switching_activity: float) -> float:
        return switching_activity * self.capacitance * (self.voltage ** 2) * self.frequency

    def calculate_static_power(self, temperature_factor: float = 1.0) -> float:
        temp_adjustment = temperature_factor * 2 ** ((self.temperature - 300) / 10)
        adjusted_leakage = self.leakage_current * temp_adjustment
        return adjusted_leakage * self.voltage

    def calculate_short_circuit_power(self, transition_time: float = 1e-10) -> float:
        transition_factor = min(transition_time / (1 / self.frequency), 0.1)
        return self.short_circuit_current * self.voltage * self.frequency * transition_factor

    def calculate_total_power(self, switching_activity: float,
                              temperature_factor: float = 1.0,
                              transition_time: float = 1e-10) -> float:
        p_dynamic = self.calculate_dynamic_power(switching_activity)
        p_static = self.calculate_static_power(temperature_factor)
        p_short_circuit = self.calculate_short_circuit_power(transition_time)
        return p_dynamic + p_static + p_short_circuit

    def hamming_weight(self, value: int) -> int:
        return bin(value).count('1')

    def hamming_distance(self, value1: int, value2: int) -> int:
        return self.hamming_weight(value1 ^ value2)

    def calculate_switching_activity(self, value1: int, value2: int,
                                     bit_weights: Optional[List[float]] = None) -> float:
        if bit_weights is None:
            bit_weights = [1.0] * self.num_bits
        transitions = value1 ^ value2
        activity = 0.0
        for bit_pos in range(self.num_bits):
            if transitions & (1 << bit_pos):
                activity += bit_weights[bit_pos]
        return activity / sum(bit_weights)

    def generate_thermal_noise(self, num_samples: int, bandwidth: float) -> np.ndarray:
        noise_power = 4 * self.k_boltzmann * self.temperature * bandwidth
        noise_voltage = np.sqrt(noise_power)
        return np.random.normal(0, noise_voltage, num_samples)

    def generate_flicker_noise(self, num_samples: int, sampling_rate: float) -> np.ndarray:
        white_noise = np.random.randn(num_samples)
        fft = np.fft.rfft(white_noise)
        freqs = np.fft.rfftfreq(num_samples, 1/sampling_rate)
        freqs[0] = freqs[1]
        fft = fft / np.sqrt(freqs)
        fft *= np.sqrt(self.flicker_noise_coefficient)
        flicker = np.fft.irfft(fft, num_samples)
        return flicker

    def calculate_snr_db(self, signal_power: float, noise_power: float) -> float:
        if noise_power <= 0:
            return float('inf')
        snr_linear = signal_power / noise_power
        return 10 * np.log10(snr_linear)

    def simulate_power_trace(self,
                             operations: List[Dict[str, any]],
                             num_samples_per_op: int = 1000,
                             sampling_rate: float = 5e9,
                             include_all_components: bool = True) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        total_samples = len(operations) * num_samples_per_op
        dynamic_power = np.zeros(total_samples)
        static_power = np.zeros(total_samples)
        short_circuit_power = np.zeros(total_samples)
        thermal_noise = np.zeros(total_samples)
        flicker_noise = np.zeros(total_samples)
        previous_value = 0
        for i, op in enumerate(operations):
            start_idx = i * num_samples_per_op
            end_idx = (i + 1) * num_samples_per_op
            if op['type'] == 'sbox':
                plaintext = op.get('plaintext', 0)
                key = op.get('key', 0)
                input_value = plaintext ^ key
                output_value = self.sbox[input_value & 0xFF]
                switching = self.calculate_switching_activity(previous_value, output_value)
                p_dyn = self.calculate_dynamic_power(switching)
                dynamic_power[start_idx:end_idx] = p_dyn
                previous_value = output_value
            elif op['type'] == 'xor':
                val1 = op.get('val1', 0)
                val2 = op.get('val2', 0)
                output_value = val1 ^ val2
                switching = self.calculate_switching_activity(previous_value, output_value)
                p_dyn = self.calculate_dynamic_power(switching)
                dynamic_power[start_idx:end_idx] = p_dyn
                previous_value = output_value
            temp_factor = 1.0 + 0.1 * np.sin(2 * np.pi * i / len(operations))
            if include_all_components:
                p_stat = self.calculate_static_power(temp_factor)
                static_power[start_idx:end_idx] = p_stat
                transition_profile = np.exp(-np.linspace(0, 5, num_samples_per_op))
                p_sc = self.calculate_short_circuit_power() * transition_profile
                short_circuit_power[start_idx:end_idx] = p_sc
        bandwidth = sampling_rate / 2
        thermal_noise = self.generate_thermal_noise(total_samples, bandwidth)
        flicker_noise = self.generate_flicker_noise(total_samples, sampling_rate)
        total_power = dynamic_power + static_power + short_circuit_power
        noisy_power = total_power + thermal_noise + flicker_noise
        signal_power = np.var(total_power)
        noise_power = np.var(thermal_noise + flicker_noise)
        snr_db = self.calculate_snr_db(signal_power, noise_power)
        components = {
            'dynamic': dynamic_power,
            'static': static_power,
            'short_circuit': short_circuit_power,
            'thermal_noise': thermal_noise,
            'flicker_noise': flicker_noise,
            'total_clean': total_power,
            'snr_db': snr_db
        }
        return noisy_power, components

    def simulate_masked_operation(self, plaintext: int, key: int, mask: int,
                                   num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        unmasked_input = plaintext ^ key
        unmasked_output = self.sbox[unmasked_input & 0xFF]
        masked_input = unmasked_input ^ mask
        masked_output = self.sbox[masked_input & 0xFF]
        output_mask = np.random.randint(0, 256)
        final_output = masked_output ^ output_mask
        ops_unmasked = [{'type': 'sbox', 'plaintext': plaintext, 'key': key}]
        ops_masked = [
            {'type': 'xor', 'val1': plaintext ^ key, 'val2': mask},
            {'type': 'sbox', 'plaintext': (plaintext ^ key) ^ mask, 'key': 0},
            {'type': 'xor', 'val1': masked_output, 'val2': output_mask}
        ]
        trace_unmasked, _ = self.simulate_power_trace(ops_unmasked, num_samples)
        trace_masked, _ = self.simulate_power_trace(ops_masked, num_samples // 3)
        return trace_unmasked, trace_masked

class PowerConsumptionModel(CompletePowerConsumptionModel):

    def __init__(self, num_bits=8, alpha=1.0, beta=1.0, sigma=0.1):
        super().__init__(
            alpha=alpha,
            num_bits=num_bits,
            thermal_noise_density=sigma**2
        )
        self.beta = beta
        self.sigma = sigma

    def calculate_power_trace(self, data, key, num_samples=1000, operations=['sbox']):
        if not isinstance(data, list):
            data = [data]
        ops = []
        for d, op in zip(data, operations):
            if op == 'sbox':
                ops.append({'type': 'sbox', 'plaintext': d, 'key': key})
            else:
                raise ValueError(f"Unknown operation: {op}")
        trace, _ = self.simulate_power_trace(ops, num_samples // len(ops))
        return trace
