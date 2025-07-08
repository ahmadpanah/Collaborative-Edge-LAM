
import numpy as np
from .lora import LoRAUpdate
from .microservice import MICROSERVICE_PORTFOLIO
from .utils import get_logger
from config import V_PARAM

class EdgeServer:
    """The central orchestrator for training, inference, and management."""
    def __init__(self, devices):
        self.logger = get_logger("EdgeServer")
        self.devices = devices
        
        # Determine max rank from all devices
        self.max_rank = max(d.lora_rank for d in devices)
        self.lora_dim = 1024 # Assumed dimension for LoRA matrices
        
        # Initialize global LoRA matrices (A, B)
        self.global_lora_A = np.zeros((self.lora_dim, self.max_rank))
        self.global_lora_B = np.zeros((self.max_rank, self.lora_dim))
        
        # Store past updates for unlearning
        self.past_updates = {}
        self.logger.info(f"Initialized with max_rank={self.max_rank}")

    # --- Heterogeneity-Aware Training ---
    
    def _dequantize(self, update: LoRAUpdate):
        """Simulates de-quantization to a common high precision (FP32)."""
        if update.precision == "INT8":
            # Simple inverse scaling for simulation
            return update.A.astype('float32') / 127.0, update.B.astype('float32') / 127.0
        return update.A.astype('float32'), update.B.astype('float32')

    def _pad_to_max_rank(self, A, B, rank):
        """Pads LoRA matrices with zeros to match the max rank."""
        pad_width_A = ((0, 0), (0, self.max_rank - rank))
        padded_A = np.pad(A, pad_width_A, 'constant')
        
        pad_width_B = ((0, self.max_rank - rank), (0, 0))
        padded_B = np.pad(B, pad_width_B, 'constant')
        return padded_A, padded_B

    def federated_training_round(self, selected_clients):
        """
        Manages one full round of heterogeneity-aware federated fine-tuning.
        Corresponds to Algorithm 1, lines 1-20.
        """
        self.logger.info(f"--- Starting Federated Training Round ---")
        updates = []
        
        # 1. Dispatch to clients
        for client in selected_clients:
            update = client.client_update(self.global_lora_A, self.global_lora_B, self.max_rank)
            updates.append(update)
            # Store update for potential future unlearning
            self.past_updates[client.id] = update

        # --- Server-side Heterogeneity-Aware Aggregation ---
        self.logger.info("Aggregating heterogeneous updates...")
        
        total_data_samples = sum(up.data_samples for up in updates)
        aggregated_A = np.zeros_like(self.global_lora_A, dtype='float32')
        aggregated_B = np.zeros_like(self.global_lora_B, dtype='float32')

        for update in updates:
            # Step 1: De-Quantize to common precision (FP32)
            dequantized_A, dequantized_B = self._dequantize(update)
            
            # Step 2: Pad to Max Rank
            padded_A, padded_B = self._pad_to_max_rank(dequantized_A, dequantized_B, update.rank)
            
            # Step 3: Perform Weighted Federated Averaging
            weight = update.data_samples / total_data_samples
            aggregated_A += padded_A * weight
            aggregated_B += padded_B * weight
            
        self.global_lora_A = aggregated_A
        self.global_lora_B = aggregated_B
        self.logger.info("Global LoRA model updated successfully.")

    # --- Precision-Aware Inference Orchestration ---
    
    def _select_optimal_pair(self, microservice_type, available_devices, qos):
        """
        Selects the best (microservice, device) pair based on QoS and system state.
        Corresponds to Algorithm 2, lines 25-46 (SELECTOPTIMALPAIR).
        """
        best_pair = None
        min_objective = float('inf')

        # Weights for Lyapunov drift-plus-penalty
        w_L = 1.0 if qos.get('priority') == 'latency' else 0.5
        w_E = 1.0 if qos.get('priority') == 'energy' else 0.5
        
        for m_service in MICROSERVICE_PORTFOLIO[microservice_type]:
            # Filter services that don't meet minimum accuracy
            if m_service.accuracy < qos['min_accuracy']:
                continue
                
            for device in available_devices:
                # Lyapunov Drift-Plus-Penalty Calculation
                latency = m_service.base_latency_ms * device.power_factor
                energy = m_service.base_energy * device.power_factor

                # Cost does not include memory, as it's a constraint, not a penalty
                performance_cost = w_L * latency + w_E * energy
                queue_cost = device.queue_length
                
                # V * PerformanceCost - QueueCost (as in paper, though often +)
                # Using a slightly modified formula for intuitive sense:
                # Balance immediate cost (perf) with long-term stability (queue)
                objective = V_PARAM * performance_cost + (1-V_PARAM) * queue_cost

                if objective < min_objective:
                    min_objective = objective
                    best_pair = (m_service, device)
        
        return best_pair

    def handle_inference_request(self, request):
        """
        Decomposes a task and orchestrates its execution across the edge network.
        Corresponds to Algorithm 2, lines 7-23.
        """
        self.logger.info(f"--- Received Inference Request: {request['task_name']} ---")
        self.logger.info(f"QoS: {request['qos']}")
        
        # Decompose request into a workflow (e.g., 4 CoT steps)
        workflow_steps = request['steps']
        total_latency = 0
        total_active_memory = 0
        
        # In a real system, this would be parallel. We simulate sequentially for clarity.
        for i, step_type in enumerate(workflow_steps):
            self.logger.info(f"Orchestrating Step {i+1}: '{step_type}'")
            
            # Find the optimal microservice and device for this step
            pair = self._select_optimal_pair(step_type, self.devices, request['qos'])
            
            if not pair:
                self.logger.error(f"Could not find a suitable device/service for step {i+1}. Aborting.")
                return None
            
            selected_service, selected_device = pair
            self.logger.info(f"Dispatching '{selected_service.name}' to Device {selected_device.id}")
            
            # Simulate execution
            exec_result = selected_device.execute_microservice(selected_service)
            total_latency += exec_result['latency']
            total_active_memory += exec_result['memory_gb']

        # The total active memory is the sum of footprints of concurrently running services.
        # For a 4-step parallel task, this is correct.
        self.logger.info(f"--- Inference Complete ---")
        return {"total_latency_ms": total_latency, "total_active_memory_gb": total_active_memory}


    # --- Federated Unlearning ---

    def federated_unlearning(self, client_id_to_forget):
        """
        Performs efficient unlearning via orthogonal projection (simulated by reversing the update).
        """
        self.logger.info(f"--- Starting Unlearning for Client {client_id_to_forget} ---")
        if client_id_to_forget not in self.past_updates:
            self.logger.error("Client has no history to unlearn. Cannot proceed.")
            return False

        # Retrieve the forgotten client's last update
        forgotten_update = self.past_updates[client_id_to_forget]
        
        # This is a simplified simulation of "projection-based unlearning".
        # It reverses the client's weighted contribution from the global model.
        # This is extremely fast compared to retraining.
        
        # 1. Dequantize and Pad the forgotten update
        dequantized_A, dequantized_B = self._dequantize(forgotten_update)
        padded_A, padded_B = self._pad_to_max_rank(dequantized_A, dequantized_B, forgotten_update.rank)
        
        # 2. Calculate the total weight and the forgotten client's weight
        # This assumes the unlearning happens right after the last round.
        total_data_samples = sum(up.data_samples for up in self.past_updates.values())
        forgotten_weight = forgotten_update.data_samples / total_data_samples

        # 3. "Subtract" the weighted contribution
        self.global_lora_A -= padded_A * forgotten_weight
        self.global_lora_B -= padded_B * forgotten_weight

        # 4. Renormalize the model
        # This is crucial to maintain the model's scale.
        self.global_lora_A /= (1 - forgotten_weight)
        self.global_lora_B /= (1 - forgotten_weight)

        self.logger.info(f"Successfully unlearned contribution from Client {client_id_to_forget}.")
        return True