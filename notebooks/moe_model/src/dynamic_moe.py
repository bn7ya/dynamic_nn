"""
Dynamic Mixture-of-Experts Model.

Extends DynamicNetwork concepts to MoE architecture:
- 4-phase training (Exploration, Estimation, Main, Fine-tuning)
- Expert efficiency tracking
- Load balancing via auxiliary loss
- Health monitoring (cancer/alzheimer for expert utilization)
- Reward/Penalty system for adaptive learning rate
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import time

try:
    from .moe_config import (
        DynamicMoEConfig,
        MoETrainingResult,
        MoEHealthReport,
    )
    from .moe_layer import MoELayer
except ImportError:
    from moe_config import (
        DynamicMoEConfig,
        MoETrainingResult,
        MoEHealthReport,
    )
    from moe_layer import MoELayer


class DynamicMoE:
    """
    Dynamic Mixture-of-Experts model extending DynamicNetwork concepts.

    Features:
    - 4-phase training (Exploration, Estimation, Main, Fine-tuning)
    - Expert efficiency tracking (like DNN node efficiency)
    - Load balancing via auxiliary loss
    - Health monitoring (cancer/alzheimer for expert utilization)
    - Reward/Penalty system for adaptive learning rate
    """

    def __init__(self, config: DynamicMoEConfig):
        self.config = config
        np.random.seed(config.seed)

        # Model dimensions
        self.vocab_size = config.vocab_size
        self.embed_dim = config.embed_dim
        self.max_seq_len = config.max_seq_len

        # Token embedding
        self.embedding = np.random.randn(config.vocab_size, config.embed_dim) * 0.02

        # MoE layer
        self.moe_layer = MoELayer(config.embed_dim, config.embed_dim, config)

        # Output projection (for next-token prediction)
        self.output_proj = np.random.randn(config.embed_dim, config.vocab_size) * 0.02
        self.output_bias = np.zeros(config.vocab_size)

        # Gradient accumulators
        self.d_embedding = np.zeros_like(self.embedding)
        self.d_output_proj = np.zeros_like(self.output_proj)
        self.d_output_bias = np.zeros_like(self.output_bias)

        # Training history
        self.cost_history: List[float] = []
        self.perplexity_history: List[float] = []
        self.efficiency_history: List[float] = []
        self.cancer_score_history: List[float] = []
        self.alzheimer_score_history: List[float] = []
        self.learning_rate_history: List[float] = []

        # Emotional state (from DNN reward/penalty system)
        self.total_rewards = 0
        self.total_penalties = 0

        # Phase tracking
        self.current_phase = 0
        self.trained = False

        # Cache
        self._cache = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through the model.

        Args:
            x: Input token IDs (batch, seq_len)

        Returns:
            Logits for next token prediction (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape

        # Token embedding
        embedded = self.embedding[x]  # (batch, seq_len, embed_dim)

        # MoE layer
        moe_out = self.moe_layer.forward(embedded, training)  # (batch, seq_len, embed_dim)

        # Output projection
        # Reshape for matmul
        moe_flat = moe_out.reshape(-1, self.embed_dim)  # (batch*seq, embed_dim)
        logits_flat = moe_flat @ self.output_proj + self.output_bias  # (batch*seq, vocab_size)
        logits = logits_flat.reshape(batch_size, seq_len, self.vocab_size)

        # Cache for backward
        if training:
            self._cache = {
                'x': x,
                'embedded': embedded,
                'moe_out': moe_out,
            }

        return logits

    def backward(self, grad_logits: np.ndarray, learning_rate: float) -> None:
        """
        Backward pass through the model.

        Args:
            grad_logits: Gradient of loss w.r.t. logits (batch, seq_len, vocab_size)
            learning_rate: Learning rate
        """
        x = self._cache['x']
        embedded = self._cache['embedded']
        moe_out = self._cache['moe_out']

        batch_size, seq_len = x.shape

        # Reshape gradients
        grad_logits_flat = grad_logits.reshape(-1, self.vocab_size)
        moe_out_flat = moe_out.reshape(-1, self.embed_dim)

        # Gradient through output projection
        self.d_output_proj = moe_out_flat.T @ grad_logits_flat / (batch_size * seq_len)
        self.d_output_bias = np.mean(grad_logits_flat, axis=0)

        grad_moe_flat = grad_logits_flat @ self.output_proj.T
        grad_moe = grad_moe_flat.reshape(batch_size, seq_len, self.embed_dim)

        # Gradient through MoE layer
        grad_embedded = self.moe_layer.backward(grad_moe, learning_rate)

        # Gradient through embedding (sparse update)
        # For each token in the input, accumulate gradient
        for b in range(batch_size):
            for s in range(seq_len):
                token_id = x[b, s]
                self.d_embedding[token_id] += grad_embedded[b, s] / (batch_size * seq_len)

        # Gradient clipping
        max_norm = self.config.max_grad_norm
        grad_norm = np.sqrt(
            np.sum(self.d_output_proj**2) +
            np.sum(self.d_output_bias**2) +
            np.sum(self.d_embedding**2)
        )
        if grad_norm > max_norm:
            scale = max_norm / (grad_norm + 1e-8)
            self.d_output_proj *= scale
            self.d_output_bias *= scale
            self.d_embedding *= scale

        # Update parameters
        self.output_proj -= learning_rate * self.d_output_proj
        self.output_bias -= learning_rate * self.d_output_bias
        self.embedding -= learning_rate * self.d_embedding

        # Reset gradients
        self.d_embedding = np.zeros_like(self.embedding)

    def compute_loss(
        self,
        logits: np.ndarray,
        targets: np.ndarray,
        include_aux_loss: bool = True
    ) -> Tuple[float, np.ndarray]:
        """
        Compute cross-entropy loss with optional auxiliary load balance loss.

        Args:
            logits: Model output (batch, seq_len, vocab_size)
            targets: Target token IDs (batch, seq_len)
            include_aux_loss: Whether to include load balance auxiliary loss

        Returns:
            Tuple of (loss, gradient of loss w.r.t. logits)
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Softmax
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)

        # Numerical stability
        logits_max = np.max(logits_flat, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_flat - logits_max)
        probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-8)

        # Cross-entropy loss
        n_samples = batch_size * seq_len
        correct_probs = probs[np.arange(n_samples), targets_flat]
        ce_loss = -np.mean(np.log(correct_probs + 1e-8))

        # Auxiliary load balance loss
        aux_loss = 0.0
        if include_aux_loss:
            aux_loss = self.moe_layer.get_load_balance_loss()

        total_loss = ce_loss + aux_loss

        # Gradient of cross-entropy w.r.t. logits
        grad_logits = probs.copy()
        grad_logits[np.arange(n_samples), targets_flat] -= 1
        grad_logits /= n_samples
        grad_logits = grad_logits.reshape(batch_size, seq_len, vocab_size)

        return total_loss, grad_logits

    def compute_perplexity(self, loss: float) -> float:
        """Compute perplexity from loss."""
        return np.exp(min(loss, 100))  # Clip to prevent overflow

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> MoETrainingResult:
        """
        Train the model using 4-phase training.

        Phase 1: Exploration - High routing noise, discover expert specializations
        Phase 2: Estimation - Estimate epochs based on efficiency convergence
        Phase 3: Main Training - Reward/penalty adaptive learning rate
        Phase 4: Fine-tuning - Frozen routing, polish weights

        Args:
            X_train: Training input (samples, seq_len)
            y_train: Training targets (samples, seq_len)
            X_val: Validation input (optional)
            y_val: Validation targets (optional)
            verbose: Whether to print progress

        Returns:
            MoETrainingResult with training metrics
        """
        start_time = time.time()

        if verbose:
            print("=" * 60)
            print("Dynamic MoE Training")
            print(f"Experts: {self.config.gating.num_experts}, Top-k: {self.config.gating.top_k}")
            print(f"Training samples: {len(X_train)}")
            print("=" * 60)

        # Phase 1: Exploration
        if verbose:
            print("\n--- Phase 1: Exploration ---")
        self._phase1_exploration(X_train, y_train, verbose)

        # Phase 2: Estimation
        if verbose:
            print("\n--- Phase 2: Estimation ---")
        estimated_epochs = self._phase2_estimation(X_train, y_train, verbose)

        # Phase 3: Main Training
        if verbose:
            print(f"\n--- Phase 3: Main Training ({estimated_epochs} epochs) ---")
        self._phase3_main_training(X_train, y_train, estimated_epochs, verbose)

        # Phase 4: Fine-tuning
        if self.config.training_phase.phase4_enabled:
            if verbose:
                print("\n--- Phase 4: Fine-tuning ---")
            self._phase4_finetuning(X_train, y_train, X_val, y_val, verbose)

        # Final evaluation
        final_loss = self._evaluate(X_val if X_val is not None else X_train,
                                    y_val if y_val is not None else y_train)

        elapsed_time = int((time.time() - start_time) * 1000)
        self.trained = True

        if verbose:
            print("\n" + "=" * 60)
            print("Training Complete")
            print(f"Final Loss: {final_loss:.4f}")
            print(f"Final Perplexity: {self.compute_perplexity(final_loss):.2f}")
            print(f"Total Time: {elapsed_time/1000:.1f}s")
            print("=" * 60)

        return MoETrainingResult(
            success=True,
            epochs_completed=len(self.cost_history),
            final_cost=final_loss,
            final_perplexity=self.compute_perplexity(final_loss),
            best_cost=min(self.cost_history) if self.cost_history else final_loss,
            stopping_reason="completed",
            cost_history=self.cost_history,
            perplexity_history=self.perplexity_history,
            efficiency_history=self.efficiency_history,
            cancer_score_history=self.cancer_score_history,
            alzheimer_score_history=self.alzheimer_score_history,
            total_rewards=self.total_rewards,
            total_penalties=self.total_penalties,
            learning_rate_history=self.learning_rate_history,
        )

    def _phase1_exploration(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool
    ) -> None:
        """
        Phase 1: Exploration with high routing noise.

        Goal: Discover which experts specialize in what patterns.
        """
        self.current_phase = 1
        self.moe_layer.set_phase(1)

        config = self.config.training_phase
        epochs = config.exploration_epochs
        lr = config.exploration_learning_rate
        batch_size = config.exploration_batch_size

        for epoch in range(epochs):
            loss = self._train_epoch(X, y, lr, batch_size)
            self.cost_history.append(loss)
            self.perplexity_history.append(self.compute_perplexity(loss))

            # Track health
            cancer, alzheimer = self.moe_layer.compute_health_scores()
            self.cancer_score_history.append(cancer)
            self.alzheimer_score_history.append(alzheimer)

            # Track efficiency
            efficiency = np.mean(self.moe_layer.get_expert_efficiencies())
            self.efficiency_history.append(efficiency)

            if verbose and (epoch + 1) % 2 == 0:
                load_dist = self.moe_layer.gating.get_load_distribution()
                print(f"  Epoch {epoch+1}/{epochs}: Loss={loss:.4f}, "
                      f"PPL={self.compute_perplexity(loss):.2f}, "
                      f"Loads={[f'{l:.2f}' for l in load_dist]}")

    def _phase2_estimation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool
    ) -> int:
        """
        Phase 2: Estimate training epochs needed.

        Uses expert efficiency convergence rate.
        """
        self.current_phase = 2
        self.moe_layer.set_phase(2)

        config = self.config.training_phase
        epochs = config.estimation_epochs
        lr = config.estimation_learning_rate
        batch_size = config.exploration_batch_size

        initial_efficiency = np.mean(self.moe_layer.get_expert_efficiencies())

        for epoch in range(epochs):
            loss = self._train_epoch(X, y, lr, batch_size)
            self.cost_history.append(loss)
            self.perplexity_history.append(self.compute_perplexity(loss))

            # Track metrics
            cancer, alzheimer = self.moe_layer.compute_health_scores()
            self.cancer_score_history.append(cancer)
            self.alzheimer_score_history.append(alzheimer)

            efficiency = np.mean(self.moe_layer.get_expert_efficiencies())
            self.efficiency_history.append(efficiency)

            if verbose and (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Loss={loss:.4f}, Efficiency={efficiency:.3f}")

        final_efficiency = np.mean(self.moe_layer.get_expert_efficiencies())

        # Estimate epochs based on efficiency improvement rate
        eff_improvement = final_efficiency - initial_efficiency

        if eff_improvement > 0:
            target_eff = config.target_efficiency
            remaining = target_eff - final_efficiency
            if remaining > 0:
                estimated = int(remaining / eff_improvement * epochs)
            else:
                estimated = config.min_estimated_epochs
        else:
            estimated = config.max_estimated_epochs // 2

        estimated = max(config.min_estimated_epochs,
                       min(config.max_estimated_epochs, estimated))

        if verbose:
            print(f"  Estimated epochs for Phase 3: {estimated}")

        return estimated

    def _phase3_main_training(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        verbose: bool
    ) -> None:
        """
        Phase 3: Main training with reward/penalty system.

        Reward: Load becomes more balanced, experts specialize
        Penalty: Load imbalance increases, experts collapse
        """
        self.current_phase = 3
        self.moe_layer.set_phase(3)

        config = self.config.training_phase
        rp_config = self.config.reward_penalty
        lr = config.main_learning_rate
        batch_size = config.main_batch_size

        prev_load_balance = self.moe_layer.gating.compute_load_balance_loss()

        for epoch in range(epochs):
            loss = self._train_epoch(X, y, lr, batch_size)
            self.cost_history.append(loss)
            self.perplexity_history.append(self.compute_perplexity(loss))
            self.learning_rate_history.append(lr)

            # Track health
            cancer, alzheimer = self.moe_layer.compute_health_scores()
            self.cancer_score_history.append(cancer)
            self.alzheimer_score_history.append(alzheimer)

            efficiency = np.mean(self.moe_layer.get_expert_efficiencies())
            self.efficiency_history.append(efficiency)

            # Reward/Penalty based on load balance improvement
            curr_load_balance = self.moe_layer.gating.compute_load_balance_loss()
            balance_improved = curr_load_balance < prev_load_balance

            # Cost improvement
            if len(self.cost_history) >= 2:
                cost_improved = (self.cost_history[-2] - self.cost_history[-1]) > \
                               rp_config.cost_improvement_threshold * self.cost_history[-2]
            else:
                cost_improved = False

            # Apply reward or penalty
            if balance_improved and cost_improved:
                lr = min(lr * rp_config.max_adjustment_factor, rp_config.max_learning_rate)
                self.total_rewards += 1
            elif not balance_improved and not cost_improved:
                lr = max(lr * rp_config.min_adjustment_factor, rp_config.min_learning_rate)
                self.total_penalties += 1

            prev_load_balance = curr_load_balance

            # Reset layer stats periodically
            if (epoch + 1) % 10 == 0:
                self.moe_layer.reset_stats()

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Loss={loss:.4f}, "
                      f"LR={lr:.5f}, R/P={self.total_rewards}/{self.total_penalties}")

    def _phase4_finetuning(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        verbose: bool
    ) -> None:
        """
        Phase 4: Fine-tuning with frozen routing.

        Only expert weights are updated, not gating network.
        """
        self.current_phase = 4
        self.moe_layer.set_phase(4)

        config = self.config.training_phase
        lr = config.phase4_learning_rate
        batch_size = config.phase4_batch_size

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(config.phase4_max_epochs):
            loss = self._train_epoch(X_train, y_train, lr, batch_size)
            self.cost_history.append(loss)
            self.perplexity_history.append(self.compute_perplexity(loss))
            self.learning_rate_history.append(lr)

            # Track metrics
            cancer, alzheimer = self.moe_layer.compute_health_scores()
            self.cancer_score_history.append(cancer)
            self.alzheimer_score_history.append(alzheimer)

            efficiency = np.mean(self.moe_layer.get_expert_efficiencies())
            self.efficiency_history.append(efficiency)

            # Validation loss for early stopping
            if X_val is not None:
                val_loss = self._evaluate(X_val, y_val)
            else:
                val_loss = loss

            # Early stopping
            if val_loss < best_loss - config.phase4_min_improvement:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= config.phase4_patience and epoch >= config.phase4_min_epochs:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

            # Learning rate decay
            if (epoch + 1) % config.phase4_lr_decay_interval == 0:
                lr = max(lr * config.phase4_lr_decay_rate, config.phase4_min_learning_rate)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: Loss={loss:.4f}, Val={val_loss:.4f}, LR={lr:.6f}")

    def _train_epoch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        learning_rate: float,
        batch_size: int
    ) -> float:
        """Train for one epoch and return average loss."""
        n_samples = len(X)
        indices = np.random.permutation(n_samples)

        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]

            X_batch = X[batch_idx]
            y_batch = y[batch_idx]

            # Forward
            logits = self.forward(X_batch, training=True)

            # Compute loss
            loss, grad_logits = self.compute_loss(logits, y_batch, include_aux_loss=True)

            # Backward
            self.backward(grad_logits, learning_rate)

            total_loss += loss
            n_batches += 1

        return total_loss / n_batches

    def _evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate on data and return average loss."""
        logits = self.forward(X, training=False)
        loss, _ = self.compute_loss(logits, y, include_aux_loss=False)
        return loss

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict next token probabilities.

        Args:
            x: Input token IDs (batch, seq_len)

        Returns:
            Probabilities for next token (batch, seq_len, vocab_size)
        """
        logits = self.forward(x, training=False)

        # Softmax
        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-8)

        return probs

    def generate(
        self,
        prompt_ids: np.ndarray,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> np.ndarray:
        """
        Generate text continuation.

        Args:
            prompt_ids: Starting token IDs (1, seq_len)
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling

        Returns:
            Generated token IDs including prompt
        """
        generated = prompt_ids.copy()

        for _ in range(max_new_tokens):
            # Get predictions for last position
            if generated.shape[1] > self.max_seq_len:
                context = generated[:, -self.max_seq_len:]
            else:
                context = generated

            probs = self.predict(context)
            next_probs = probs[0, -1, :]  # Last position

            # Apply temperature
            if temperature != 1.0:
                logits = np.log(next_probs + 1e-8) / temperature
                logits_max = np.max(logits)
                exp_logits = np.exp(logits - logits_max)
                next_probs = exp_logits / np.sum(exp_logits)

            # Top-k sampling
            if top_k > 0:
                top_indices = np.argsort(next_probs)[-top_k:]
                mask = np.zeros_like(next_probs)
                mask[top_indices] = 1
                next_probs = next_probs * mask
                next_probs = next_probs / (np.sum(next_probs) + 1e-8)

            # Sample
            next_token = np.random.choice(len(next_probs), p=next_probs)

            # Append
            generated = np.concatenate([generated, [[next_token]]], axis=1)

            # Stop at EOS (id=3)
            if next_token == 3:
                break

        return generated

    def get_health_report(self) -> MoEHealthReport:
        """Get current health report for the model."""
        return self.moe_layer.get_health_report()

    def get_expert_stats(self) -> Dict[str, Any]:
        """Get detailed expert statistics."""
        return self.moe_layer.get_stats()
