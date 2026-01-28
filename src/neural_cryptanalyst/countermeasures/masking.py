import numpy as np
from typing import Optional, Tuple

class MaskingCountermeasure:

    def apply_boolean_masking(self, data: np.ndarray,
                              mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:

        if mask is None:
            mask = np.random.randint(0, 256, size=data.shape, dtype=data.dtype)
        return data ^ mask, mask
