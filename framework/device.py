import numpy as np
import time
from .lora import LoRAUpdate
from .utils import get_logger

class EdgeDevice:
    """Simulates a heterogeneous edge device."""
    def __init__(self, device_id, tier, rank, precision, power_factor, quantization_fn):
        self.id = device_id
        self.tier = tier
        self.lora_rank = rank
        self.precision = precision
        self.power_factor = power_factor
        self.quantization_fn = quantization_fn
        
        self.logger = get_logger(f"Device-{self.id} ({self.tier})")
        self.queue_length = 0 # Represents current load
        self.local_data_samples = np.random.randint(50, 200) # Simulate non-IID data

    def client_update(self, global_lora_A, global_lora_B, max_rank):
        """
        Simulates one round of federated fine-tuning on the client.
        Corresponds to Algorithm 1, lines 22-32 (CLIENTUPDATE).
        """
        self.logger.info(f"Starting training round. Global rank={max_rank}, my rank={self.lora_rank}")

        # 1. Truncate global model to device's rank
        # (Alocal, Blocal) <- TruncateToRank(Ag, Bg, rk)
        local_A = global_lora_A[:, :self.lora_rank]
        local_B = global_lora_B[:self.lora_rank, :]
        
        # 2. Simulate local training loop
        # This is a placeholder for actual training.
        self.logger.info("Simulating local epoch training...")
        time.sleep(0.1 * self.power_factor) # Training takes longer on weaker devices
        
        # Simulate generating a local update
        # In reality, this comes from backpropagation.
        local_update_A = np.random.randn(*local_A.shape) * 0.01
        local_update_B = np.random.randn(*local_B.shape) * 0.01
        
        # 3. Apply hardware-specific quantization (QAT simulation)
        # Lk(Qk(Wo + BlocalAlocal); b)
        quantized_A = self.quantization_fn(local_update_A)
        quantized_B = self.quantization_fn(local_update_B)
        self.logger.info(f"Quantized local update to {self.precision}")

        return LoRAUpdate(
            client_id=self.id,
            A=quantized_A,
            B=quantized_B,
            rank=self.lora_rank,
            precision=self.precision,
            data_samples=self.local_data_samples
        )

    def execute_microservice(self, microservice):
        """Simulates executing a microservice on this device."""
        latency = microservice.base_latency_ms * self.power_factor
        energy = microservice.base_energy * self.power_factor
        
        self.logger.info(f"Executing '{microservice.name}'. Estimated latency: {latency:.2f}ms")
        self.queue_length += latency # Add to device queue
        
        # Simulate work
        time.sleep(latency / 1000)
        
        self.queue_length -= latency # Remove from queue after completion
        return {"latency": latency, "energy": energy, "memory_gb": microservice.memory_gb}