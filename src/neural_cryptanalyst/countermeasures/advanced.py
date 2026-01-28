import numpy as np
from typing import List, Tuple, Optional, Callable, Dict
import random
import time

class BlindingCountermeasures:

    def __init__(self, rng_seed: Optional[int] = None):
        self.rng = np.random.RandomState(rng_seed)
        random.seed(rng_seed)

    def rsa_blinding(self, message: int, d: int, n: int, e: int) -> int:
        while True:
            r = self.rng.randint(2, n - 1)
            if np.gcd(r, n) == 1:
                break
        r_e = pow(r, e, n)
        blinded_message = (message * r_e) % n
        blinded_result = pow(blinded_message, d, n)
        r_inv = pow(r, -1, n)
        result = (blinded_result * r_inv) % n
        expected = pow(message, d, n)
        assert result == expected, "Blinding failed!"
        return result

    def _mod_inverse(self, a: int, m: int) -> int:

        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y

        gcd, x, _ = extended_gcd(a % m, m)
        if gcd != 1:
            raise ValueError("Modular inverse does not exist")
        return (x % m + m) % m

    def ecc_scalar_blinding(self, scalar: int, point, curve_order: int):
        r = self.rng.randint(1, 100)
        blinded_scalar = scalar + r * curve_order
        result = f"EC_POINT(blinded_scalar={blinded_scalar})"
        return result

    def aes_masking_with_shuffling(
        self, state: List[int], key: List[int], sbox: List[int]
    ) -> Tuple[List[int], List[int]]:
        masks_in = [self.rng.randint(0, 256) for _ in range(16)]
        masks_out = [self.rng.randint(0, 256) for _ in range(16)]
        masked_sbox = [0] * 256
        for i in range(256):
            masked_input = i
            unmasked_input = masked_input ^ masks_in[0]
            masked_output = sbox[unmasked_input] ^ masks_out[0]
            masked_sbox[masked_input] = masked_output
        shuffle_order = list(range(16))
        random.shuffle(shuffle_order)
        output = [0] * 16
        for idx in shuffle_order:
            masked_byte = state[idx] ^ key[idx] ^ masks_in[idx % len(masks_in)]
            masked_output = masked_sbox[masked_byte]
            output[idx] = masked_output
        return output, shuffle_order

    def hiding_with_dummy_operations(
        self, real_operation: Callable, num_dummies: int = 5
    ) -> any:
        real_position = self.rng.randint(0, num_dummies + 1)
        result = None
        for i in range(num_dummies + 1):
            if i == real_position:
                result = real_operation()
            else:
                _ = real_operation()
        return result

    def randomized_exponentiation(self, base: int, exponent: int, modulus: int) -> int:
        random_factor = self.rng.randint(0, 10)
        algorithm = self.rng.choice(["binary", "sliding_window", "montgomery"])
        if algorithm == "binary":
            return self._binary_exponentiation(base, exponent, modulus)
        elif algorithm == "sliding_window":
            return self._sliding_window_exponentiation(base, exponent, modulus)
        else:
            return self._montgomery_exponentiation(base, exponent, modulus)

    def _binary_exponentiation(self, base: int, exp: int, mod: int) -> int:
        result = 1
        base = base % mod
        while exp > 0:
            if exp % 2 == 1:
                result = (result * base) % mod
            exp = exp >> 1
            base = (base * base) % mod
        return result

    def _sliding_window_exponentiation(
        self, base: int, exp: int, mod: int, window_size: int = 4
    ) -> int:
        precomputed = [1]
        power = base % mod
        for _ in range(1, 2 ** window_size):
            precomputed.append((precomputed[-1] * power) % mod)
        exp_binary = bin(exp)[2:]
        result = 1
        i = 0
        while i < len(exp_binary):
            if exp_binary[i] == "0":
                result = (result * result) % mod
                i += 1
            else:
                j = min(i + window_size, len(exp_binary))
                while j > i and exp_binary[j - 1] == "0":
                    j -= 1
                window = int(exp_binary[i:j], 2)
                for _ in range(j - i):
                    result = (result * result) % mod
                result = (result * precomputed[window]) % mod
                i = j
        return result

    def _montgomery_exponentiation(self, base: int, exp: int, mod: int) -> int:
        r0 = 1
        r1 = base % mod
        for bit in bin(exp)[2:]:
            if bit == "0":
                r1 = (r0 * r1) % mod
                r0 = (r0 * r0) % mod
            else:
                r0 = (r0 * r1) % mod
                r1 = (r1 * r1) % mod
        return r0

class RandomizationTechniques:

    def __init__(self, rng_seed: Optional[int] = None):
        self.rng = np.random.RandomState(rng_seed)

    def random_delays(
        self, operation: Callable, min_delay_us: int = 0, max_delay_us: int = 1000
    ) -> any:
        delay = self.rng.randint(min_delay_us, max_delay_us + 1)
        time.sleep(delay / 1_000_000)
        return operation()

    def operation_shuffling(
        self, operations: List[Callable], dependencies: Optional[List[Tuple[int, int]]] = None
    ) -> List[any]:
        n = len(operations)
        if dependencies is None:
            order = list(range(n))
            self.rng.shuffle(order)
        else:
            order = self._randomized_topological_sort(n, dependencies)
        results = [None] * n
        for idx in order:
            results[idx] = operations[idx]()
        return results

    def _randomized_topological_sort(
        self, n: int, dependencies: List[Tuple[int, int]]
    ) -> List[int]:
        adj = [[] for _ in range(n)]
        in_degree = [0] * n
        for i, j in dependencies:
            adj[i].append(j)
            in_degree[j] += 1
        queue = [i for i in range(n) if in_degree[i] == 0]
        result = []
        while queue:
            idx = self.rng.randint(0, len(queue))
            node = queue.pop(idx)
            result.append(node)
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        return result

    def clock_randomization(self, operation: Callable, jitter_percent: float = 10.0) -> any:
        scale = 1.0 + (self.rng.random() - 0.5) * jitter_percent / 100
        start_time = time.time()
        result = operation()
        base_duration = time.time() - start_time
        jitter_delay = base_duration * (scale - 1.0)
        if jitter_delay > 0:
            time.sleep(jitter_delay)
        return result

    def power_balancing(self, sensitive_value: int, num_bits: int = 8) -> Tuple[int, int]:
        mask = (1 << num_bits) - 1
        complement = (~sensitive_value) & mask
        if self.rng.random() < 0.5:
            _ = bin(sensitive_value).count("1")
            _ = bin(complement).count("1")
            return sensitive_value, complement
        else:
            _ = bin(complement).count("1")
            _ = bin(sensitive_value).count("1")
            return sensitive_value, complement

class IntegratedCountermeasures:

    def __init__(self, rng_seed: Optional[int] = None):
        self.blinding = BlindingCountermeasures(rng_seed)
        self.randomization = RandomizationTechniques(rng_seed)
        self.rng = np.random.RandomState(rng_seed)

    def protected_aes_round(
        self,
        state: List[int],
        round_key: List[int],
        sbox: List[int],
        protection_level: int = 3,
    ) -> List[int]:
        if protection_level >= 1:
            mask = self.rng.randint(0, 256, 16).tolist()
            masked_state = [s ^ m for s, m in zip(state, mask)]
        else:
            masked_state = state
            mask = [0] * 16
        if protection_level >= 2:
            result, shuffle_order = self.blinding.aes_masking_with_shuffling(
                masked_state, round_key, sbox
            )
        else:
            result = []
            for i in range(16):
                result.append(sbox[(masked_state[i] ^ round_key[i]) & 0xFF])
        if protection_level >= 3:
            num_dummies = self.rng.randint(2, 5)

            def dummy_sbox():
                dummy_input = self.rng.randint(0, 256)
                return sbox[dummy_input]

            for _ in range(num_dummies):
                self.randomization.random_delays(dummy_sbox, 0, 100)
        if protection_level >= 1:
            result = [r ^ m for r, m in zip(result, mask)]
        return result

    def protected_rsa_decryption(
        self,
        ciphertext: int,
        d: int,
        n: int,
        e: int,
        protection_level: int = 3,
    ) -> int:
        if protection_level >= 1:
            if protection_level >= 3:
                result = self.randomization.random_delays(
                    lambda: self.blinding.rsa_blinding(ciphertext, d, n, e),
                    0,
                    1000,
                )
            else:
                result = self.blinding.rsa_blinding(ciphertext, d, n, e)
        else:
            result = pow(ciphertext, d, n)
        if protection_level >= 2:
            check = self.blinding.randomized_exponentiation(ciphertext, d, n)
            assert result == check, "Computation verification failed"
        return result

    def evaluate_protection_overhead(
        self,
        operation: Callable,
        protection_level: int = 3,
        num_trials: int = 100,
    ) -> Dict:
        import time

        start = time.time()
        for _ in range(num_trials):
            operation()
        baseline_time = time.time() - start
        protected_times = {}
        for level in range(1, protection_level + 1):
            start = time.time()
            for _ in range(num_trials):
                if level >= 1:
                    _ = self.rng.randint(0, 256, 16)
                if level >= 2:
                    order = list(range(16))
                    self.rng.shuffle(order)
                if level >= 3:
                    time.sleep(0.0001)
                operation()
            protected_times[level] = time.time() - start
        overheads = {}
        for level, prot_time in protected_times.items():
            overhead_percent = ((prot_time - baseline_time) / baseline_time) * 100
            overheads[f"level_{level}_overhead"] = overhead_percent
        return {
            "baseline_time": baseline_time,
            "protected_times": protected_times,
            "overheads": overheads,
            "num_trials": num_trials,
        }
