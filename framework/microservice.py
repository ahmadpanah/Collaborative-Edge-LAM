from dataclasses import dataclass

@dataclass
class Microservice:
    """Represents a virtualized, multi-precision LAM function."""
    name: str
    precision: str
    accuracy: float         # Model accuracy for this variant (0.0 to 1.0)
    base_latency_ms: int    # Latency on a high-tier device
    base_energy: float      # Energy on a high-tier device
    memory_gb: float        # Memory footprint

# The portfolio of available microservices for an 'expert' or 'CoT_step'
# This simulates the different versions mentioned in the paper.
MICROSERVICE_PORTFOLIO = {
    "CoT_Step": [
        Microservice(
            name="CoT_Step_FP16", precision="FP16", accuracy=0.99,
            base_latency_ms=100, base_energy=10, memory_gb=1.2
        ),
        Microservice(
            name="CoT_Step_INT8", precision="INT8", accuracy=0.96,
            base_latency_ms=40, base_energy=4, memory_gb=0.4
        ),
        Microservice(
            name="CoT_Step_Pruned_INT8", precision="Pruned_INT8", accuracy=0.92,
            base_latency_ms=25, base_energy=2.5, memory_gb=0.25
        ),
    ]
}