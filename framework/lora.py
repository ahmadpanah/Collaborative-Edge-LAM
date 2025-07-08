from dataclasses import dataclass
import numpy as np

@dataclass
class LoRAUpdate:
    """Represents a LoRA update from a client."""
    client_id: int
    A: np.ndarray
    B: np.ndarray
    rank: int
    precision: str
    data_samples: int # Used for weighted averaging