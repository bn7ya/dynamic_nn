"""
MoE Layer implementation.

Combines experts and gating network with sparse routing
and health monitoring.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional

try:
    from .moe_config import DynamicMoEConfig, MoEHealthReport, DynamicExpertConfig
    from .expert import Expert
    from .gating import GatingNetwork
except ImportError:
    from moe_config import DynamicMoEConfig, MoEHealthReport, DynamicExpertConfig
    from expert import Expert
    from gating import GatingNetwork


class MoELayer:
    """
    Mixture-of-Experts layer with dynamic efficiency tracking.

    Features:
    - Sparse routing (only compute selected experts)
    - Weighted combination of expert outputs
    - Health monitoring (cancer/alzheimer scores)
    - Efficiency tracking per expert
    """

    def __init__(self, input_dim: int, output_dim: int, config: DynamicMoEConfig):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config
        self.num_experts = config.gating.num_experts
        self.top_k = config.gating.top_k

        # Initialize gating network
        self.gating = GatingNetwork(
            input_dim,
            config.gating,
            config.load_balance,
            seed=config.seed
        )

        # Initialize experts
        self.experts: List[Expert] = []
        for i in range(self.num_experts):
            expert = Expert(
                input_dim,
                output_dim,
                config.expert,
                expert_id=i,
                seed=config.seed + i
            )
            self.experts.append(expert)

        # Health tracking
        self.cancer_score_history: List[float] = []
        self.alzheimer_score_history: List[float] = []
        self.health_reports: List[MoEHealthReport] = []

        # Cache for backward pass
        self._cache = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through MoE layer.

        Args:
            x: Input tensor (batch, input_dim) or (batch, seq_len, input_dim)

        Returns:
            Output tensor with same shape as input (except last dim is output_dim)
        """
        # Handle 3D input (batch, seq_len, dim)
        original_shape = x.shape
        if len(original_shape) == 3:
            batch_size, seq_len, _ = original_shape
            x_flat = x.reshape(-1, self.input_dim)  # (batch*seq, input_dim)
        else:
            batch_size = x.shape[0]
            seq_len = 1
            x_flat = x

        num_tokens = x_flat.shape[0]

        # Get routing weights
        router_weights, expert_indices, router_probs = self.gating.forward(x_flat, training)

        # Initialize output
        output = np.zeros((num_tokens, self.output_dim))

        # Store per-expert outputs for backward pass
        expert_outputs = {}
        expert_inputs = {}

        # Compute expert outputs (sparse - only for selected experts)
        for i in range(self.num_experts):
            # Find which tokens use this expert (in any of the top_k positions)
            mask = np.zeros(num_tokens, dtype=bool)
            weights_for_expert = np.zeros(num_tokens)

            for k in range(self.top_k):
                k_mask = expert_indices[:, k] == i
                mask |= k_mask
                weights_for_expert[k_mask] = router_weights[k_mask, k]

            if not np.any(mask):
                continue

            # Get expert output for these tokens
            x_expert = x_flat[mask]
            expert_out = self.experts[i].forward(x_expert, training)

            # Store for backward
            if training:
                expert_outputs[i] = expert_out
                expert_inputs[i] = mask

            # Weight by router probability and accumulate
            weights = weights_for_expert[mask][:, np.newaxis]
            output[mask] += weights * expert_out

        # Cache for backward
        if training:
            self._cache = {
                'x_flat': x_flat,
                'router_weights': router_weights,
                'expert_indices': expert_indices,
                'router_probs': router_probs,
                'expert_outputs': expert_outputs,
                'expert_inputs': expert_inputs,
                'original_shape': original_shape,
            }

        # Reshape output if needed
        if len(original_shape) == 3:
            output = output.reshape(batch_size, seq_len, self.output_dim)

        return output

    def backward(self, grad_output: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backward pass through MoE layer.

        Args:
            grad_output: Gradient of loss w.r.t. output
            learning_rate: Learning rate

        Returns:
            Gradient of loss w.r.t. input
        """
        original_shape = self._cache['original_shape']
        x_flat = self._cache['x_flat']
        router_weights = self._cache['router_weights']
        expert_indices = self._cache['expert_indices']
        expert_outputs = self._cache['expert_outputs']
        expert_inputs = self._cache['expert_inputs']

        # Flatten gradient if needed
        if len(original_shape) == 3:
            grad_output = grad_output.reshape(-1, self.output_dim)

        num_tokens = grad_output.shape[0]
        grad_input = np.zeros_like(x_flat)

        # Gradient for router weights
        grad_router_weights = np.zeros_like(router_weights)

        # Backward through experts
        for i in range(self.num_experts):
            if i not in expert_inputs:
                continue

            mask = expert_inputs[i]
            expert_out = expert_outputs[i]

            # Weights for this expert
            weights_for_expert = np.zeros(num_tokens)
            for k in range(self.top_k):
                k_mask = expert_indices[:, k] == i
                weights_for_expert[k_mask] = router_weights[k_mask, k]

            weights = weights_for_expert[mask][:, np.newaxis]

            # Gradient w.r.t. expert output: d_loss/d_expert_out = weights * d_loss/d_output
            grad_expert_out = grad_output[mask] * weights

            # Backward through expert
            grad_expert_input = self.experts[i].backward(grad_expert_out, learning_rate)
            grad_input[mask] += grad_expert_input

            # Gradient w.r.t. weights: d_loss/d_weight = expert_out * d_loss/d_output
            # Sum over output dimension
            grad_weight = np.sum(expert_out * grad_output[mask], axis=-1)

            # Scatter back to router_weights gradient
            for k in range(self.top_k):
                k_mask = expert_indices[:, k] == i
                combined_mask = mask.copy()
                combined_mask[~k_mask] = False  # Only tokens where this expert is at position k
                grad_router_weights[combined_mask, k] = grad_weight[combined_mask[mask]]

        # Backward through gating network
        self.gating.backward(grad_router_weights, learning_rate)

        # Reshape gradient if needed
        if len(original_shape) == 3:
            batch_size, seq_len, _ = original_shape
            grad_input = grad_input.reshape(batch_size, seq_len, self.input_dim)

        return grad_input

    def compute_health_scores(self) -> Tuple[float, float]:
        """
        Compute MoE-specific health scores.

        Cancer: Expert domination (one expert getting too much traffic)
        Alzheimer: Expert death (expert not being used)
        """
        cancer_score = self.gating.compute_cancer_score()
        alzheimer_score = self.gating.compute_alzheimer_score()

        self.cancer_score_history.append(cancer_score)
        self.alzheimer_score_history.append(alzheimer_score)

        return cancer_score, alzheimer_score

    def get_expert_efficiencies(self) -> np.ndarray:
        """Get efficiency scores for all experts."""
        return np.array([expert.compute_efficiency() for expert in self.experts])

    def get_load_balance_loss(self) -> float:
        """Get auxiliary load balance loss."""
        return self.gating.compute_load_balance_loss()

    def get_health_report(self) -> MoEHealthReport:
        """Generate comprehensive health report."""
        cancer, alzheimer = self.compute_health_scores()
        overall_health = 1.0 - max(cancer, alzheimer)

        # Determine state
        if overall_health > self.config.health.healthy_threshold:
            state = 'healthy'
        elif overall_health > 1.0 - self.config.health.at_risk_threshold:
            state = 'at_risk'
        else:
            state = 'critical'

        # Get expert metrics
        expert_loads = self.gating.get_load_distribution().tolist()
        expert_efficiencies = self.get_expert_efficiencies().tolist()

        # Generate diagnosis
        diagnosis_parts = []
        recommendations = []

        if cancer > 0.3:
            dominant_idx = np.argmax(self.gating.get_load_distribution())
            diagnosis_parts.append(f"Expert {dominant_idx} is dominating with {expert_loads[dominant_idx]:.1%} traffic")
            recommendations.append("Increase load balance weight to encourage diversity")
            recommendations.append("Add more routing noise during training")

        if alzheimer > 0.3:
            dead_experts = [i for i, load in enumerate(expert_loads)
                           if load < self.config.load_balance.min_expert_load]
            diagnosis_parts.append(f"Experts {dead_experts} are underutilized")
            recommendations.append("Increase exploration noise")
            recommendations.append("Consider reducing number of experts")

        if not diagnosis_parts:
            diagnosis_parts.append("Load balance is healthy")

        report = MoEHealthReport(
            state=state,
            cancer_score=cancer,
            alzheimer_score=alzheimer,
            overall_health=overall_health,
            expert_loads=expert_loads,
            expert_efficiencies=expert_efficiencies,
            diagnosis="; ".join(diagnosis_parts),
            recommendations=recommendations,
        )

        self.health_reports.append(report)
        return report

    def set_phase(self, phase: int) -> None:
        """Configure layer for training phase."""
        self.gating.set_noise_for_phase(phase)

        if phase == 4:
            self.gating.freeze()
        else:
            self.gating.unfreeze()

    def reset_stats(self) -> None:
        """Reset statistics (called between epochs)."""
        self.gating.reset_load_stats()
        for expert in self.experts:
            expert.reset_recent_stats()

    def get_stats(self) -> dict:
        """Get layer statistics for monitoring."""
        expert_stats = [expert.get_stats() for expert in self.experts]
        gating_stats = self.gating.get_stats()

        return {
            'experts': expert_stats,
            'gating': gating_stats,
            'mean_expert_efficiency': np.mean(self.get_expert_efficiencies()),
            'cancer_score': self.cancer_score_history[-1] if self.cancer_score_history else 0,
            'alzheimer_score': self.alzheimer_score_history[-1] if self.alzheimer_score_history else 0,
        }

    # ==================== Dynamic Expert Modification ====================

    def _add_expert(self, config: DynamicExpertConfig) -> bool:
        """
        Add a new expert by cloning from the most-loaded expert.

        Process:
        1. Check max_experts constraint
        2. Find most-loaded expert (highest utilization)
        3. Clone expert with noise
        4. Expand gating network
        5. Append to expert list

        Args:
            config: Dynamic expert configuration

        Returns:
            bool: True if expert was added, False if at max capacity
        """
        # Check constraint
        if len(self.experts) >= config.max_experts:
            return False

        # Find most-loaded expert
        load_dist = self.gating.get_load_distribution()
        most_loaded_idx = int(np.argmax(load_dist))
        source_expert = self.experts[most_loaded_idx]

        # Create new expert slot in gating
        new_expert_idx = self.gating.add_expert_slot()

        # Clone the most-loaded expert with noise
        new_expert = Expert.clone_with_noise(
            source_expert=source_expert,
            new_expert_id=new_expert_idx,
            noise_scale=config.clone_noise_scale,
            seed=self.config.seed + new_expert_idx + len(self.experts) * 100
        )

        # Add to expert list
        self.experts.append(new_expert)

        # Update layer state
        self.num_experts = len(self.experts)

        return True

    def _remove_expert(self, idx: int, config: DynamicExpertConfig) -> bool:
        """
        Remove an expert by index.

        Process:
        1. Check min_experts constraint
        2. Validate index
        3. Remove from expert list
        4. Contract gating network
        5. Re-index remaining experts

        Args:
            idx: Index of expert to remove
            config: Dynamic expert configuration

        Returns:
            bool: True if expert was removed, False if at minimum or invalid index
        """
        # Check constraint - must keep at least min_experts
        if len(self.experts) <= config.min_experts:
            return False

        # Validate index
        if idx < 0 or idx >= len(self.experts):
            return False

        # Remove from gating network first
        self.gating.remove_expert_slot(idx)

        # Remove from expert list
        del self.experts[idx]

        # Re-index remaining experts
        for new_idx, expert in enumerate(self.experts):
            expert.expert_id = new_idx

        # Update layer state
        self.num_experts = len(self.experts)

        # Clear cache since expert indices changed
        self._cache = {}

        return True

    def _merge_experts(
        self,
        idx1: int,
        idx2: int,
        config: DynamicExpertConfig
    ) -> bool:
        """
        Merge two experts into one.

        Process:
        1. Check min_experts constraint (merge reduces count by 1)
        2. Validate indices
        3. Create merged expert
        4. Remove both source experts from gating
        5. Add merged expert slot
        6. Update expert list

        Args:
            idx1: Index of first expert to merge
            idx2: Index of second expert to merge
            config: Dynamic expert configuration

        Returns:
            bool: True if merge succeeded, False otherwise
        """
        # After merge we have (N-1) experts, must maintain min_experts
        if len(self.experts) - 1 < config.min_experts:
            return False

        # Validate indices
        if idx1 == idx2:
            return False
        if idx1 < 0 or idx1 >= len(self.experts):
            return False
        if idx2 < 0 or idx2 >= len(self.experts):
            return False

        # Ensure idx1 < idx2 for removal order
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1

        expert1 = self.experts[idx1]
        expert2 = self.experts[idx2]

        # Create merged expert
        merged_expert = Expert.merge_experts(
            expert1=expert1,
            expert2=expert2,
            merged_expert_id=idx1,  # Will take the lower index
            efficiency_weight=config.merge_efficiency_weight,
            seed=self.config.seed + len(self.experts) * 100
        )

        # Remove higher index first (to preserve lower index)
        self.gating.remove_expert_slot(idx2)
        del self.experts[idx2]

        # Replace expert at idx1 with merged expert
        self.experts[idx1] = merged_expert

        # Re-index all experts
        for new_idx, expert in enumerate(self.experts):
            expert.expert_id = new_idx

        # Update layer state
        self.num_experts = len(self.experts)

        # Clear cache
        self._cache = {}

        return True

    def _should_modify_architecture(
        self,
        epoch: int,
        config: DynamicExpertConfig
    ) -> Tuple[str, Optional[int], Optional[int]]:
        """
        Determine if architecture should be modified this epoch.

        Uses efficiency-based decisions similar to DynamicNetwork.

        Checks:
        1. Addition trigger: High load imbalance (cancer) OR high overall efficiency
        2. Removal trigger: Expert with low importance score
        3. Merge trigger: Two experts with very similar weights

        Args:
            epoch: Current epoch number
            config: Dynamic expert configuration

        Returns:
            Tuple of (action, idx1, idx2) where:
            - action: 'none', 'add', 'remove', or 'merge'
            - idx1: Primary index (for remove/merge)
            - idx2: Secondary index (for merge only)
        """
        # Only check at intervals
        if epoch % config.architecture_change_interval != 0:
            return ('none', None, None)

        # Get current metrics
        cancer_score = self.gating.compute_cancer_score()
        alzheimer_score = self.gating.compute_alzheimer_score()
        efficiencies = self.get_expert_efficiencies()
        mean_efficiency = np.mean(efficiencies)
        load_dist = self.gating.get_load_distribution()

        # --- Check for ADDITION ---
        # Trigger: Cancer (expert domination) OR very high efficiency (model can handle more)
        if config.enable_expert_addition:
            should_add = (
                (cancer_score > config.add_load_imbalance_threshold) or
                (mean_efficiency > config.add_efficiency_threshold)
            )
            if should_add and np.random.random() < config.add_probability:
                if len(self.experts) < config.max_experts:
                    return ('add', None, None)

        # --- Check for REMOVAL ---
        # Trigger: Expert with low importance (low utilization + gradients + loss contribution)
        if config.enable_expert_removal:
            if len(self.experts) > config.min_experts:
                # Compute importance scores for all experts
                importance_scores = []
                for expert in self.experts:
                    importance = expert.compute_importance(
                        utilization_weight=config.efficiency_utilization_weight,
                        gradient_weight=config.efficiency_gradient_weight,
                        loss_weight=config.efficiency_loss_weight
                    )
                    importance_scores.append(importance)

                # Find least important expert
                min_importance_idx = int(np.argmin(importance_scores))
                min_importance = importance_scores[min_importance_idx]

                # Also check load - if very low utilization, candidate for removal
                min_load_idx = int(np.argmin(load_dist))
                min_load = load_dist[min_load_idx]

                # Remove if importance is very low OR utilization is very low
                should_remove = (
                    (min_importance < config.remove_efficiency_threshold) or
                    (min_load < config.remove_utilization_threshold)
                )

                if should_remove and np.random.random() < config.remove_probability:
                    # Choose the expert to remove (prefer low importance)
                    remove_idx = min_importance_idx
                    return ('remove', remove_idx, None)

        # --- Check for MERGE ---
        # Trigger: Two experts with very similar weights
        if config.enable_expert_merging:
            if len(self.experts) > config.min_experts:
                # Compute pairwise weight similarity
                weight_vectors = [expert.get_weight_vector() for expert in self.experts]

                max_similarity = -1.0
                merge_pair = (None, None)

                for i in range(len(self.experts)):
                    for j in range(i + 1, len(self.experts)):
                        # Cosine similarity of weight vectors
                        v1, v2 = weight_vectors[i], weight_vectors[j]
                        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
                        if norm1 > 1e-8 and norm2 > 1e-8:
                            similarity = np.dot(v1, v2) / (norm1 * norm2)
                        else:
                            similarity = 0.0

                        if similarity > max_similarity:
                            max_similarity = similarity
                            merge_pair = (i, j)

                if max_similarity > config.merge_similarity_threshold:
                    if np.random.random() < config.merge_probability:
                        return ('merge', merge_pair[0], merge_pair[1])

        return ('none', None, None)
