"""
Evaluation utilities for Dynamic MoE model.

Provides metrics, comparison, and analysis tools.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple

try:
    from .dynamic_moe import DynamicMoE
    from .moe_config import DynamicMoEConfig, MoETrainingResult
except ImportError:
    from dynamic_moe import DynamicMoE
    from moe_config import DynamicMoEConfig, MoETrainingResult


class MoEEvaluator:
    """
    Evaluator for Dynamic MoE model.

    Features:
    - Standard NLP metrics (perplexity, accuracy)
    - Expert utilization analysis
    - Load balance metrics
    - Comparison with baseline models
    """

    def __init__(self, model: DynamicMoE):
        self.model = model

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 64
    ) -> Dict[str, float]:
        """
        Evaluate model on data.

        Args:
            X: Input token IDs
            y: Target token IDs
            batch_size: Batch size for evaluation

        Returns:
            Dictionary with evaluation metrics
        """
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        n_batches = 0

        n_samples = len(X)

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch = X[start:end]
            y_batch = y[start:end]

            # Forward pass
            logits = self.model.forward(X_batch, training=False)

            # Loss
            loss, _ = self.model.compute_loss(logits, y_batch, include_aux_loss=False)
            total_loss += loss

            # Accuracy
            predictions = np.argmax(logits, axis=-1)
            # Mask padding tokens (id=0)
            mask = y_batch != 0
            correct = (predictions == y_batch) & mask
            total_correct += np.sum(correct)
            total_tokens += np.sum(mask)

            n_batches += 1

        avg_loss = total_loss / n_batches
        accuracy = total_correct / max(1, total_tokens)
        perplexity = self.model.compute_perplexity(avg_loss)

        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'accuracy': accuracy,
            'total_tokens': total_tokens,
        }

    def analyze_experts(self) -> Dict[str, Any]:
        """
        Analyze expert behavior and specialization.

        Returns:
            Dictionary with expert analysis
        """
        stats = self.model.get_expert_stats()
        gating = stats['gating']
        experts = stats['experts']

        # Load distribution analysis
        loads = np.array(gating['expert_loads'])
        load_entropy = gating['load_entropy']
        uniform_entropy = np.log(len(loads))
        load_uniformity = load_entropy / uniform_entropy

        # Expert health
        cancer = gating['cancer_score']
        alzheimer = gating['alzheimer_score']

        # Per-expert metrics
        expert_metrics = []
        for i, exp in enumerate(experts):
            expert_metrics.append({
                'id': i,
                'load': loads[i],
                'efficiency': exp['efficiency'],
                'utilization': exp['utilization'],
                'gradient_magnitude': exp['gradient_magnitude'],
                'dead_nodes': exp['dead_nodes'],
                'status': self._classify_expert(loads[i], exp['efficiency']),
            })

        # Find dominant and underutilized experts
        dominant = [e for e in expert_metrics if e['status'] == 'dominant']
        underutilized = [e for e in expert_metrics if e['status'] == 'underutilized']
        healthy = [e for e in expert_metrics if e['status'] == 'healthy']

        return {
            'load_distribution': loads.tolist(),
            'load_uniformity': load_uniformity,
            'load_balance_loss': gating['load_balance_loss'],
            'cancer_score': cancer,
            'alzheimer_score': alzheimer,
            'expert_metrics': expert_metrics,
            'num_dominant': len(dominant),
            'num_underutilized': len(underutilized),
            'num_healthy': len(healthy),
            'mean_efficiency': stats['mean_expert_efficiency'],
        }

    def _classify_expert(self, load: float, efficiency: float) -> str:
        """Classify expert status based on load and efficiency."""
        num_experts = self.model.config.gating.num_experts
        expected_load = 1.0 / num_experts

        if load > expected_load * 2:
            return 'dominant'
        elif load < expected_load * 0.2:
            return 'underutilized'
        elif efficiency < 0.3:
            return 'low_efficiency'
        else:
            return 'healthy'

    def compute_expert_specialization(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_samples: int = 1000
    ) -> Dict[str, Any]:
        """
        Analyze what each expert specializes in.

        This examines which tokens/patterns each expert handles.
        """
        # Sample data
        if len(X) > num_samples:
            indices = np.random.choice(len(X), num_samples, replace=False)
            X = X[indices]
            y = y[indices]

        # Track which expert handles which tokens
        expert_token_counts = [np.zeros(self.model.vocab_size)
                               for _ in range(self.model.config.gating.num_experts)]

        # Forward pass to get routing decisions
        batch_size = 64
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            X_batch = X[start:end]

            # Get embeddings
            embedded = self.model.embedding[X_batch]

            # Get routing
            x_flat = embedded.reshape(-1, self.model.embed_dim)
            _, expert_indices, _ = self.model.moe_layer.gating.forward(x_flat, training=False)

            # Track which experts handle which tokens
            tokens_flat = X_batch.reshape(-1)
            for i, token_id in enumerate(tokens_flat):
                for k in range(self.model.config.gating.top_k):
                    expert_idx = expert_indices[i, k]
                    expert_token_counts[expert_idx][token_id] += 1

        # Find top tokens per expert
        expert_specializations = []
        for i, counts in enumerate(expert_token_counts):
            top_indices = np.argsort(counts)[-10:][::-1]
            total = np.sum(counts)
            top_tokens = [
                {'token_id': int(idx), 'count': int(counts[idx]),
                 'fraction': counts[idx] / max(1, total)}
                for idx in top_indices if counts[idx] > 0
            ]
            expert_specializations.append({
                'expert_id': i,
                'total_tokens': int(total),
                'top_tokens': top_tokens,
            })

        return {
            'expert_specializations': expert_specializations,
        }

    def compare_with_baseline(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        baseline_type: str = 'single_expert'
    ) -> Dict[str, Any]:
        """
        Compare MoE model with a baseline.

        Args:
            X_test: Test input
            y_test: Test targets
            baseline_type: Type of baseline ('single_expert', 'average')

        Returns:
            Comparison metrics
        """
        # MoE performance
        moe_metrics = self.evaluate(X_test, y_test)

        # Create baseline
        if baseline_type == 'single_expert':
            # Use only the first expert (no routing)
            baseline_metrics = self._evaluate_single_expert(X_test, y_test, expert_idx=0)
        else:
            # Average all expert outputs (no gating)
            baseline_metrics = self._evaluate_average_experts(X_test, y_test)

        # Compute improvements
        loss_improvement = (baseline_metrics['loss'] - moe_metrics['loss']) / baseline_metrics['loss']
        ppl_improvement = (baseline_metrics['perplexity'] - moe_metrics['perplexity']) / baseline_metrics['perplexity']
        acc_improvement = (moe_metrics['accuracy'] - baseline_metrics['accuracy']) / max(0.01, baseline_metrics['accuracy'])

        return {
            'moe': moe_metrics,
            'baseline': baseline_metrics,
            'baseline_type': baseline_type,
            'improvements': {
                'loss': loss_improvement,
                'perplexity': ppl_improvement,
                'accuracy': acc_improvement,
            }
        }

    def _evaluate_single_expert(
        self,
        X: np.ndarray,
        y: np.ndarray,
        expert_idx: int
    ) -> Dict[str, float]:
        """Evaluate using only a single expert."""
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        n_batches = 0
        batch_size = 64

        expert = self.model.moe_layer.experts[expert_idx]

        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            X_batch = X[start:end]
            y_batch = y[start:end]

            # Embedding
            embedded = self.model.embedding[X_batch]

            # Single expert forward
            batch_size_curr, seq_len, embed_dim = embedded.shape
            x_flat = embedded.reshape(-1, embed_dim)
            expert_out = expert.forward(x_flat, training=False)
            moe_out = expert_out.reshape(batch_size_curr, seq_len, -1)

            # Output projection
            moe_flat = moe_out.reshape(-1, self.model.embed_dim)
            logits_flat = moe_flat @ self.model.output_proj + self.model.output_bias
            logits = logits_flat.reshape(batch_size_curr, seq_len, -1)

            # Loss
            loss, _ = self.model.compute_loss(logits, y_batch, include_aux_loss=False)
            total_loss += loss

            # Accuracy
            predictions = np.argmax(logits, axis=-1)
            mask = y_batch != 0
            correct = (predictions == y_batch) & mask
            total_correct += np.sum(correct)
            total_tokens += np.sum(mask)

            n_batches += 1

        avg_loss = total_loss / n_batches
        return {
            'loss': avg_loss,
            'perplexity': self.model.compute_perplexity(avg_loss),
            'accuracy': total_correct / max(1, total_tokens),
        }

    def _evaluate_average_experts(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate using average of all experts (no routing)."""
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        n_batches = 0
        batch_size = 64

        num_experts = len(self.model.moe_layer.experts)

        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            X_batch = X[start:end]
            y_batch = y[start:end]

            # Embedding
            embedded = self.model.embedding[X_batch]

            # Average all expert outputs
            batch_size_curr, seq_len, embed_dim = embedded.shape
            x_flat = embedded.reshape(-1, embed_dim)

            avg_out = np.zeros((x_flat.shape[0], self.model.embed_dim))
            for expert in self.model.moe_layer.experts:
                expert_out = expert.forward(x_flat, training=False)
                avg_out += expert_out / num_experts

            moe_out = avg_out.reshape(batch_size_curr, seq_len, -1)

            # Output projection
            moe_flat = moe_out.reshape(-1, self.model.embed_dim)
            logits_flat = moe_flat @ self.model.output_proj + self.model.output_bias
            logits = logits_flat.reshape(batch_size_curr, seq_len, -1)

            # Loss
            loss, _ = self.model.compute_loss(logits, y_batch, include_aux_loss=False)
            total_loss += loss

            # Accuracy
            predictions = np.argmax(logits, axis=-1)
            mask = y_batch != 0
            correct = (predictions == y_batch) & mask
            total_correct += np.sum(correct)
            total_tokens += np.sum(mask)

            n_batches += 1

        avg_loss = total_loss / n_batches
        return {
            'loss': avg_loss,
            'perplexity': self.model.compute_perplexity(avg_loss),
            'accuracy': total_correct / max(1, total_tokens),
        }

    def generate_report(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> str:
        """
        Generate a comprehensive evaluation report.

        Returns:
            Formatted report string
        """
        # Collect metrics
        eval_metrics = self.evaluate(X_test, y_test)
        expert_analysis = self.analyze_experts()
        comparison = self.compare_with_baseline(X_test, y_test)

        # Format report
        lines = [
            "=" * 60,
            "Dynamic MoE Evaluation Report",
            "=" * 60,
            "",
            "Model Configuration:",
            f"  - Experts: {self.model.config.gating.num_experts}",
            f"  - Top-k: {self.model.config.gating.top_k}",
            f"  - Embedding dim: {self.model.embed_dim}",
            f"  - Vocab size: {self.model.vocab_size}",
            "",
            "Evaluation Metrics:",
            f"  - Loss: {eval_metrics['loss']:.4f}",
            f"  - Perplexity: {eval_metrics['perplexity']:.2f}",
            f"  - Accuracy: {eval_metrics['accuracy']:.2%}",
            "",
            "Expert Analysis:",
            f"  - Load Uniformity: {expert_analysis['load_uniformity']:.2%}",
            f"  - Mean Efficiency: {expert_analysis['mean_efficiency']:.3f}",
            f"  - Cancer Score: {expert_analysis['cancer_score']:.3f}",
            f"  - Alzheimer Score: {expert_analysis['alzheimer_score']:.3f}",
            f"  - Healthy Experts: {expert_analysis['num_healthy']}/{self.model.config.gating.num_experts}",
            "",
            "Load Distribution:",
        ]

        for i, load in enumerate(expert_analysis['load_distribution']):
            status = expert_analysis['expert_metrics'][i]['status']
            lines.append(f"  Expert {i}: {load:.2%} ({status})")

        lines.extend([
            "",
            "Comparison with Single Expert Baseline:",
            f"  - Baseline Loss: {comparison['baseline']['loss']:.4f}",
            f"  - MoE Loss: {comparison['moe']['loss']:.4f}",
            f"  - Loss Improvement: {comparison['improvements']['loss']:.1%}",
            f"  - Perplexity Improvement: {comparison['improvements']['perplexity']:.1%}",
            "",
            "=" * 60,
        ])

        return "\n".join(lines)
