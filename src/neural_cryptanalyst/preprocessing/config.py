from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, Dict, Any
import json

@dataclass
class PreprocessingConfig:

    sampling_rate: float = 1e9
    dc_offset: bool = True
    detrend_type: Optional[str] = 'linear'
    align_method: str = 'correlation'
    filter_type: Optional[str] = 'lowpass'
    filter_cutoff: Optional[Union[float, Tuple[float, float]]] = None
    filter_order: int = 5
    standardize_method: str = 'robust'
    target_length: Optional[int] = None
    max_shift: Optional[int] = None
    outlier_threshold: float = 3.0
    noise_level: float = 0.0
    augmentation_factor: int = 1

    def __post_init__(self) -> None:
        if self.sampling_rate <= 0:
            raise ValueError(f"Invalid sampling_rate {self.sampling_rate}")
        if self.filter_order < 1:
            raise ValueError("filter_order must be >= 1")
        if self.detrend_type not in ('linear', 'constant', None):
            raise ValueError(f"Invalid detrend_type {self.detrend_type}")
        if self.align_method not in ('correlation', 'peak', 'dtw', None):
            raise ValueError(f"Invalid align_method {self.align_method}")
        if self.standardize_method not in ('robust', 'standard', 'minmax'):
            raise ValueError(f"Invalid standardize_method {self.standardize_method}")
        if self.filter_cutoff is None and self.filter_type == 'lowpass':
            self.filter_cutoff = 0.4 * self.sampling_rate

    def to_dict(self) -> Dict[str, Any]:
        return {
            'sampling_rate': self.sampling_rate,
            'dc_offset': self.dc_offset,
            'detrend_type': self.detrend_type,
            'align_method': self.align_method,
            'filter_type': self.filter_type,
            'filter_cutoff': self.filter_cutoff,
            'filter_order': self.filter_order,
            'standardize_method': self.standardize_method,
            'target_length': self.target_length,
            'max_shift': self.max_shift,
            'outlier_threshold': self.outlier_threshold,
            'noise_level': self.noise_level,
            'augmentation_factor': self.augmentation_factor,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreprocessingConfig":
        return cls(**data)

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PreprocessingConfig":
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
