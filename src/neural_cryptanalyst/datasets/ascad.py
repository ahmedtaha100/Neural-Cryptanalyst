import h5py
import numpy as np
from typing import Tuple, Optional

class ASCADDataset:

    def __init__(self):
        self.traces = None
        self.labels = None
        self.metadata = None
        self.plaintexts = None
        self.target_byte_index = 2

    def load_ascad_v1(self, filepath: str, target_byte: int = 2,
                      load_plaintexts: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        self.target_byte_index = target_byte

        with h5py.File(filepath, 'r') as f:
            self.traces = np.array(f['Profiling_traces/traces'])
            self.metadata = np.array(f['Profiling_traces/metadata'])

            keys = self.metadata['key']
            if load_plaintexts:
                self.plaintexts = self.metadata['plaintext']

            from ..utils.crypto import aes_sbox

            n_traces = len(self.traces)
            self.labels = np.zeros(n_traces, dtype=np.uint8)

            for i in range(n_traces):
                if keys.ndim == 1:
                    key_byte = keys[i] if np.isscalar(keys[i]) else keys[i][target_byte]
                else:
                    key_byte = keys[i, target_byte]

                if self.plaintexts is not None:
                    if self.plaintexts.ndim == 1:
                        pt_byte = self.plaintexts[i]
                    else:
                        pt_byte = self.plaintexts[i, target_byte]
                    self.labels[i] = aes_sbox(int(pt_byte) ^ int(key_byte))
                else:
                    self.labels[i] = key_byte

        return self.traces, self.labels

    def get_attack_set(self, filepath: str,
                       target_byte: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if target_byte is None:
            target_byte = self.target_byte_index

        with h5py.File(filepath, 'r') as f:
            attack_traces = np.array(f['Attack_traces/traces'])
            attack_metadata = np.array(f['Attack_traces/metadata'])

            from ..utils.crypto import aes_sbox

            keys = attack_metadata['key']
            plaintexts = attack_metadata['plaintext']

            n_traces = len(attack_traces)
            attack_labels = np.zeros(n_traces, dtype=np.uint8)

            for i in range(n_traces):
                if keys.ndim == 1:
                    key_byte = keys[i] if np.isscalar(keys[i]) else keys[i][target_byte]
                else:
                    key_byte = keys[i, target_byte]

                if plaintexts.ndim == 1:
                    pt_byte = plaintexts[i]
                else:
                    pt_byte = plaintexts[i, target_byte]

                attack_labels[i] = aes_sbox(int(pt_byte) ^ int(key_byte))

        return attack_traces, attack_labels, attack_metadata

    def get_correct_key(self, filepath: str, target_byte: Optional[int] = None) -> int:
        if target_byte is None:
            target_byte = self.target_byte_index

        with h5py.File(filepath, 'r') as f:
            metadata = np.array(f['Attack_traces/metadata'])
            keys = metadata['key']
            if keys.ndim == 1:
                return int(keys[0]) if np.isscalar(keys[0]) else int(keys[0][target_byte])
            return int(keys[0, target_byte])
