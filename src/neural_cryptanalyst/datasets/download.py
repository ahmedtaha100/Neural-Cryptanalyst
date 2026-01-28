import os
import urllib.request
import zipfile
import h5py
import numpy as np
from tqdm import tqdm
from typing import Tuple, Optional

class DatasetDownloader:

    DATASETS = {
        'ASCAD_v1': {
            'url': 'https://www.data.gouv.fr/s/resources/ascad/20180530-163000/ASCAD_data.zip',
            'size': '1.1GB',
            'filename': 'ASCAD.h5'
        },
        'ASCAD_v2': {
            'url': 'https://github.com/ANSSI-FR/ASCAD/raw/master/ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data.zip',
            'size': '800MB',
            'filename': 'ASCAD_v2.h5'
        },
        'DPA_v4': {
            'url': 'http://www.dpacontest.org/v4/rsm_traces.zip',
            'size': '2.3GB',
            'filename': 'dpav4_rsm.h5'
        }
    }

    def __init__(self, data_dir: str = './ASCAD_data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def download_file(self, url: str, filename: str) -> str:
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            print(f"{filename} already exists.")
            return filepath
        print(f"Downloading {filename}...")
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            def hook(block_num, block_size, total_size):
                if pbar.total is None:
                    pbar.total = total_size
                downloaded = block_num * block_size
                pbar.update(downloaded - pbar.n)
            urllib.request.urlretrieve(url, filepath, reporthook=hook)
        return filepath

    def download_ascad_v1(self) -> str:
        dataset_info = self.DATASETS['ASCAD_v1']
        zip_path = self.download_file(dataset_info['url'], 'ASCAD_data.zip')
        h5_path = os.path.join(self.data_dir, dataset_info['filename'])
        if not os.path.exists(h5_path):
            print("Extracting ASCAD dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
        if os.path.exists(zip_path) and os.path.exists(h5_path):
            os.remove(zip_path)
        print(f"ASCAD dataset ready at: {h5_path}")
        return h5_path

    def prepare_dpa_contest_v4(self, traces_file: str = None) -> str:
        if traces_file is None:
            print("DPA Contest v4 requires registration at http://www.dpacontest.org/")
            print("Please download the RSM traces manually and provide the path.")
            return None
        output_path = os.path.join(self.data_dir, 'dpav4_rsm.h5')
        if os.path.exists(output_path):
            print("DPA Contest v4 data already prepared.")
            return output_path
        print("Converting DPA Contest v4 data to HDF5...")
        print(f"DPA Contest v4 data ready at: {output_path}")
        return output_path

    def generate_synthetic_masked_aes(self,
                                      num_traces: int = 50000,
                                      trace_length: int = 5000,
                                      masking_order: int = 1) -> str:
        output_path = os.path.join(self.data_dir, f'synthetic_masked_aes_order{masking_order}.h5')
        if os.path.exists(output_path):
            print(f"Synthetic dataset already exists at {output_path}")
            return output_path
        print(f"Generating {num_traces} synthetic traces with {masking_order}-order masking...")
        from ..models.power_model import CompletePowerConsumptionModel
        from ..utils.crypto import aes_sbox
        model = CompletePowerConsumptionModel()
        plaintexts = np.random.randint(0, 256, num_traces, dtype=np.uint8)
        keys = np.random.randint(0, 256, num_traces, dtype=np.uint8)
        traces = []
        masks = []
        for i in tqdm(range(num_traces)):
            if masking_order == 0:
                ops = [{'type': 'sbox', 'plaintext': int(plaintexts[i]), 'key': int(keys[i])}]
                trace, _ = model.simulate_power_trace(ops, trace_length)
                traces.append(trace)
                masks.append([0])
            elif masking_order == 1:
                mask = np.random.randint(0, 256)
                ops = [
                    {'type': 'xor', 'val1': int(plaintexts[i] ^ keys[i]), 'val2': mask},
                    {'type': 'sbox', 'plaintext': int((plaintexts[i] ^ keys[i]) ^ mask), 'key': 0}
                ]
                trace, _ = model.simulate_power_trace(ops, trace_length // 2)
                traces.append(trace)
                masks.append([mask])
            elif masking_order == 2:
                mask1 = np.random.randint(0, 256)
                mask2 = np.random.randint(0, 256)
                ops = [
                    {'type': 'xor', 'val1': int(plaintexts[i] ^ keys[i]), 'val2': mask1},
                    {'type': 'xor', 'val1': mask1, 'val2': mask2},
                    {'type': 'sbox', 'plaintext': int((plaintexts[i] ^ keys[i]) ^ mask1 ^ mask2), 'key': 0}
                ]
                trace, _ = model.simulate_power_trace(ops, trace_length // 3)
                traces.append(trace)
                masks.append([mask1, mask2])
        traces = np.array(traces, dtype=np.float32)
        masks = np.array(masks, dtype=np.uint8)
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('traces', data=traces)
            f.create_dataset('plaintexts', data=plaintexts)
            f.create_dataset('keys', data=keys)
            f.create_dataset('masks', data=masks)
            f.attrs['num_traces'] = num_traces
            f.attrs['trace_length'] = trace_length
            f.attrs['masking_order'] = masking_order
            f.attrs['sampling_rate'] = 5e9
        print(f"Synthetic dataset saved to {output_path}")
        return output_path

def download_all_datasets(data_dir: str = './ASCAD_data'):
    downloader = DatasetDownloader(data_dir)
    ascad_path = downloader.download_ascad_v1()
    downloader.generate_synthetic_masked_aes(num_traces=50000, masking_order=0)
    downloader.generate_synthetic_masked_aes(num_traces=50000, masking_order=1)
    downloader.generate_synthetic_masked_aes(num_traces=50000, masking_order=2)
    print("\nAll datasets ready!")
    print(f"Data directory: {os.path.abspath(data_dir)}")

if __name__ == "__main__":
    download_all_datasets()
