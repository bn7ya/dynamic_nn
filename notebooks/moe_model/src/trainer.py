"""
Training wrapper for Dynamic MoE model.

Provides a high-level interface for training with progress tracking
and visualization.
"""

import numpy as np
from typing import Optional, Dict, Any, List
import time

try:
    from .moe_config import DynamicMoEConfig, MoETrainingResult
    from .dynamic_moe import DynamicMoE
    from .tokenizer import SimpleTokenizer
    from .data_loader import SlimPajamaLoader
except ImportError:
    from moe_config import DynamicMoEConfig, MoETrainingResult
    from dynamic_moe import DynamicMoE
    from tokenizer import SimpleTokenizer
    from data_loader import SlimPajamaLoader


class MoETrainer:
    """
    High-level trainer for Dynamic MoE model.

    Features:
    - Easy setup with data loading and tokenization
    - Training with progress bars (if tqdm available)
    - Checkpoint saving and loading
    - Training visualization
    """

    def __init__(
        self,
        config: Optional[DynamicMoEConfig] = None,
        verbose: bool = True
    ):
        self.config = config or DynamicMoEConfig()
        self.verbose = verbose

        self.model: Optional[DynamicMoE] = None
        self.tokenizer: Optional[SimpleTokenizer] = None
        self.data_loader: Optional[SlimPajamaLoader] = None

        self.training_result: Optional[MoETrainingResult] = None

        # Check for tqdm
        try:
            from tqdm import tqdm
            self._tqdm_available = True
        except ImportError:
            self._tqdm_available = False

    def setup(
        self,
        data_source: str = 'synthetic',
        subset_size: int = 5000,
        vocab_size: int = 10000,
        max_seq_len: int = 64
    ) -> None:
        """
        Set up model, tokenizer, and data loader.

        Args:
            data_source: 'slimpajama' or 'synthetic'
            subset_size: Number of samples to use
            vocab_size: Vocabulary size
            max_seq_len: Maximum sequence length
        """
        if self.verbose:
            print("Setting up MoE Trainer...")

        # Update config
        self.config.vocab_size = vocab_size
        self.config.max_seq_len = max_seq_len

        # Initialize tokenizer
        self.tokenizer = SimpleTokenizer(vocab_size=vocab_size)

        # Initialize data loader
        self.data_loader = SlimPajamaLoader(
            subset_size=subset_size,
            max_seq_len=max_seq_len,
            tokenizer=self.tokenizer,
            seed=self.config.seed
        )

        # Load data (this also builds vocabulary)
        if self.verbose:
            print(f"Loading data ({data_source})...")

        if data_source == 'slimpajama':
            self.X, self.y = self.data_loader.load_dataset(build_vocab=True)
        else:
            self.X, self.y = self.data_loader._generate_synthetic_data(build_vocab=True)

        # Update vocab size to actual size
        actual_vocab = self.tokenizer.actual_vocab_size
        self.config.vocab_size = actual_vocab

        # Initialize model
        self.model = DynamicMoE(self.config)

        if self.verbose:
            print(f"Setup complete:")
            print(f"  - Vocabulary: {actual_vocab} tokens")
            print(f"  - Training samples: {len(self.X)}")
            print(f"  - Sequence length: {max_seq_len}")
            print(f"  - Experts: {self.config.gating.num_experts}")
            print(f"  - Top-k: {self.config.gating.top_k}")

    def train(
        self,
        val_ratio: float = 0.1
    ) -> MoETrainingResult:
        """
        Train the model.

        Args:
            val_ratio: Fraction of data for validation

        Returns:
            Training result with metrics
        """
        if self.model is None:
            raise RuntimeError("Call setup() before train()")

        # Split data
        X_train, y_train, X_val, y_val = self.data_loader.train_val_split(
            self.X, self.y, val_ratio=val_ratio
        )

        # Train
        self.training_result = self.model.fit(
            X_train, y_train,
            X_val, y_val,
            verbose=self.verbose
        )

        return self.training_result

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on data.

        Returns:
            Dictionary with loss, perplexity, and accuracy
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        # Get predictions
        logits = self.model.forward(X, training=False)

        # Compute loss
        loss, _ = self.model.compute_loss(logits, y, include_aux_loss=False)
        perplexity = self.model.compute_perplexity(loss)

        # Compute accuracy (next-token prediction)
        predictions = np.argmax(logits, axis=-1)
        accuracy = np.mean(predictions == y)

        return {
            'loss': loss,
            'perplexity': perplexity,
            'accuracy': accuracy,
        }

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling

        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized")

        # Encode prompt
        prompt_ids = self.tokenizer.encode(
            prompt,
            max_length=self.config.max_seq_len,
            add_special_tokens=True,
            padding=False
        )
        prompt_ids = prompt_ids.reshape(1, -1)

        # Generate
        generated_ids = self.model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )

        # Decode
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def get_expert_analysis(self) -> Dict[str, Any]:
        """
        Get detailed expert analysis.

        Returns:
            Dictionary with expert loads, efficiencies, and health metrics
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        stats = self.model.get_expert_stats()
        health = self.model.get_health_report()

        return {
            'expert_loads': stats['gating']['expert_loads'],
            'expert_efficiencies': [e['efficiency'] for e in stats['experts']],
            'mean_efficiency': stats['mean_expert_efficiency'],
            'cancer_score': health.cancer_score,
            'alzheimer_score': health.alzheimer_score,
            'health_state': health.state,
            'diagnosis': health.diagnosis,
            'recommendations': health.recommendations,
        }

    def plot_training_curves(self) -> None:
        """Plot training curves (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for plotting")
            return

        if self.training_result is None:
            print("No training results to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Loss curve
        ax = axes[0, 0]
        ax.plot(self.training_result.cost_history)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.grid(True)

        # Perplexity
        ax = axes[0, 1]
        ax.plot(self.training_result.perplexity_history)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Perplexity')
        ax.set_title('Perplexity')
        ax.grid(True)

        # Health scores
        ax = axes[1, 0]
        ax.plot(self.training_result.cancer_score_history, label='Cancer')
        ax.plot(self.training_result.alzheimer_score_history, label='Alzheimer')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Health Scores')
        ax.legend()
        ax.grid(True)

        # Efficiency
        ax = axes[1, 1]
        ax.plot(self.training_result.efficiency_history)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Efficiency')
        ax.set_title('Mean Expert Efficiency')
        ax.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_expert_utilization(self) -> None:
        """Plot expert utilization (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for plotting")
            return

        if self.model is None:
            print("Model not initialized")
            return

        stats = self.model.get_expert_stats()
        loads = stats['gating']['expert_loads']
        efficiencies = [e['efficiency'] for e in stats['experts']]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Load distribution
        ax = axes[0]
        x = range(len(loads))
        ax.bar(x, loads, color='steelblue')
        ax.axhline(y=1/len(loads), color='r', linestyle='--', label='Uniform')
        ax.set_xlabel('Expert')
        ax.set_ylabel('Load')
        ax.set_title('Expert Load Distribution')
        ax.set_xticks(x)
        ax.legend()

        # Efficiency
        ax = axes[1]
        ax.bar(x, efficiencies, color='forestgreen')
        ax.axhline(y=0.5, color='r', linestyle='--', label='Threshold')
        ax.set_xlabel('Expert')
        ax.set_ylabel('Efficiency')
        ax.set_title('Expert Efficiency')
        ax.set_xticks(x)
        ax.legend()

        plt.tight_layout()
        plt.show()

    def save_model(self, path: str) -> None:
        """Save model weights."""
        if self.model is None:
            raise RuntimeError("Model not initialized")

        np.savez(
            path,
            embedding=self.model.embedding,
            output_proj=self.model.output_proj,
            output_bias=self.model.output_bias,
            # Expert weights
            **{f'expert_{i}_W1': e.W1 for i, e in enumerate(self.model.moe_layer.experts)},
            **{f'expert_{i}_b1': e.b1 for i, e in enumerate(self.model.moe_layer.experts)},
            **{f'expert_{i}_W2': e.W2 for i, e in enumerate(self.model.moe_layer.experts)},
            **{f'expert_{i}_b2': e.b2 for i, e in enumerate(self.model.moe_layer.experts)},
            # Gating weights
            W_gate=self.model.moe_layer.gating.W_gate,
            b_gate=self.model.moe_layer.gating.b_gate,
        )
        print(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load model weights."""
        if self.model is None:
            raise RuntimeError("Model not initialized")

        data = np.load(path)

        self.model.embedding = data['embedding']
        self.model.output_proj = data['output_proj']
        self.model.output_bias = data['output_bias']

        for i, expert in enumerate(self.model.moe_layer.experts):
            expert.W1 = data[f'expert_{i}_W1']
            expert.b1 = data[f'expert_{i}_b1']
            expert.W2 = data[f'expert_{i}_W2']
            expert.b2 = data[f'expert_{i}_b2']

        self.model.moe_layer.gating.W_gate = data['W_gate']
        self.model.moe_layer.gating.b_gate = data['b_gate']

        print(f"Model loaded from {path}")
