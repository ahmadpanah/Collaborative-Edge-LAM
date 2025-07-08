# --- Model & Task Configuration ---
BASE_MODEL_NAME = "Qwen2.5-7B-Instruct"
BASE_MODEL_MEMORY_GB = 14.5  # Full model memory footprint
COT_TASK_STEPS = 4  # Number of steps for a Chain-of-Thought task

# --- Device Tiers Configuration ---
DEVICE_TIERS = {
    "high": {
        "count": 2,
        "lora_rank": 16,
        "precision": "FP16",
        "power_factor": 1.0,  # Baseline performance
        "quantization_fn": lambda x: x.astype('float16')
    },
    "mid": {
        "count": 4,
        "lora_rank": 8,
        "precision": "FP16",
        "power_factor": 1.5, # Takes 1.5x longer than high-tier
        "quantization_fn": lambda x: x.astype('float16')
    },
    "low": {
        "count": 4,
        "lora_rank": 4,
        "precision": "INT8", # This device uses INT8
        "power_factor": 4.0, # Takes 4x longer than high-tier
        "quantization_fn": lambda x: (x * 127).clip(-128, 127).astype('int8')
    }
}
# Total number of devices
NUM_DEVICES = sum(tier["count"] for tier in DEVICE_TIERS.values())


# --- Federated Learning Configuration ---
FEDERATED_ROUNDS = 50
COMM_OVERHEAD_PER_PARAM_LORA = 4 # bytes for a 32-bit float
FULL_MODEL_PARAMS = 7 * 10**9 # 7 Billion parameters

# --- Inference Orchestration (Lyapunov) Configuration ---
V_PARAM = 0.5  # Lyapunov trade-off parameter (balances perf vs. stability)

# --- Network Simulation ---
NETWORK_RTT_MS = 80 # Cloud round-trip-time

# --- Unlearning ---
UNLEARNING_TIME_MINUTES = 2
FULL_RETRAINING_TIME_HOURS = 5