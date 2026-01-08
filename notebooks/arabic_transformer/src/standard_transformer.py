"""
Standard Transformer Implementation

A pure Python transformer with fixed architecture for comparison with
the Dynamic NN-enhanced version. Implements a classification head
for conversational response prediction.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

from .transformer_components import (
    Embedding,
    PositionalEncoding,
    LayerNorm,
    TransformerEncoderLayer
)


@dataclass
class TransformerConfig:
    """Configuration for Standard Transformer"""
    vocab_size: int = 1000
    max_seq_len: int = 32
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    ffn_dim: int = 512
    dropout: float = 0.1
    num_classes: int = 100
    seed: int = 42


class StandardTransformer:
    """
    Standard (non-dynamic) Transformer for classification.
    Architecture is fixed throughout training.

    Used as baseline for comparison with DynamicTransformer.
    """

    def __init__(self, config: TransformerConfig):
        self.config = config
        np.random.seed(config.seed)

        # Embedding layers
        self.token_embedding = Embedding(config.vocab_size, config.embed_dim, config.seed)
        self.pos_encoding = PositionalEncoding(config.max_seq_len, config.embed_dim)

        # Encoder layers
        self.layers = [
            TransformerEncoderLayer(
                config.embed_dim,
                config.num_heads,
                config.ffn_dim,
                config.dropout,
                config.seed + i
            )
            for i in range(config.num_layers)
        ]

        # Final layer norm
        self.final_norm = LayerNorm(config.embed_dim)

        # Classification head
        self.classifier = np.random.randn(config.embed_dim, config.num_classes) * np.sqrt(2.0 / config.embed_dim)
        self.classifier_bias = np.zeros(config.num_classes)

        # Gradients for classifier
        self.grad_classifier = np.zeros_like(self.classifier)
        self.grad_classifier_bias = np.zeros_like(self.classifier_bias)

        # Training state
        self.trained = False
        self.training_history: Dict = {}

        # Cache
        self._cache: Dict = {}

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-8)

    def forward(
        self,
        token_ids: np.ndarray,
        training: bool = True
    ) -> np.ndarray:
        """
        Forward pass.

        Args:
            token_ids: (batch_size, seq_len) integer IDs

        Returns:
            logits: (batch_size, num_classes)
        """
        # Embedding + positional
        x = self.token_embedding.forward(token_ids)
        x = self.pos_encoding.forward(x)

        # Encoder layers
        for layer in self.layers:
            x = layer.forward(x, training=training)

        # Final layer norm
        x = self.final_norm.forward(x)

        # Pool (mean over sequence, ignoring padding)
        # For simplicity, we use mean pooling over all positions
        pooled = np.mean(x, axis=1)  # (batch_size, embed_dim)

        self._cache['pooled'] = pooled
        self._cache['token_ids'] = token_ids

        # Classification
        logits = pooled @ self.classifier + self.classifier_bias

        return logits

    def backward(self, grad_logits: np.ndarray) -> None:
        """
        Backward pass through entire network.

        Args:
            grad_logits: (batch_size, num_classes) gradient from loss
        """
        pooled = self._cache['pooled']
        batch_size = pooled.shape[0]
        seq_len = self._cache['token_ids'].shape[1]

        # Gradient through classifier
        self.grad_classifier = pooled.T @ grad_logits
        self.grad_classifier_bias = grad_logits.sum(axis=0)
        grad_pooled = grad_logits @ self.classifier.T

        # Gradient through mean pooling
        grad_x = np.repeat(grad_pooled[:, np.newaxis, :], seq_len, axis=1) / seq_len

        # Gradient through final layer norm
        grad_x = self.final_norm.backward(grad_x)

        # Gradient through encoder layers (in reverse order)
        for layer in reversed(self.layers):
            grad_x = layer.backward(grad_x)

        # Gradient through embedding
        self.token_embedding.backward(grad_x)

    def _cross_entropy_loss(
        self,
        logits: np.ndarray,
        targets: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Compute cross-entropy loss and gradient.

        Args:
            logits: (batch_size, num_classes) raw logits
            targets: (batch_size,) integer class labels

        Returns:
            loss: scalar loss value
            grad: (batch_size, num_classes) gradient
        """
        batch_size = logits.shape[0]

        # Softmax
        probs = self._softmax(logits)

        # Cross-entropy loss
        log_probs = np.log(probs + 1e-8)
        loss = -np.mean(log_probs[np.arange(batch_size), targets])

        # Gradient
        grad = probs.copy()
        grad[np.arange(batch_size), targets] -= 1
        grad /= batch_size

        return loss, grad

    def zero_grad(self) -> None:
        """Reset all gradients"""
        self.token_embedding.zero_grad()
        self.final_norm.zero_grad()
        for layer in self.layers:
            layer.zero_grad()
        self.grad_classifier.fill(0)
        self.grad_classifier_bias.fill(0)

    def _clip_gradients(self, max_norm: float = 1.0) -> float:
        """Clip gradients by global norm"""
        all_grads = []

        all_grads.extend(self.token_embedding.gradients())
        all_grads.extend(self.final_norm.gradients())
        for layer in self.layers:
            all_grads.extend(layer.gradients())
        all_grads.extend([self.grad_classifier, self.grad_classifier_bias])

        # Compute global norm
        total_norm = 0.0
        for grad in all_grads:
            total_norm += np.sum(grad ** 2)
        total_norm = np.sqrt(total_norm)

        # Clip if needed
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-8)
            for grad in all_grads:
                grad *= scale

        return total_norm

    def _update_parameters(self, learning_rate: float) -> None:
        """Update parameters using gradients"""
        # Update embedding
        for param, grad in zip(self.token_embedding.parameters(), self.token_embedding.gradients()):
            param -= learning_rate * grad

        # Update final norm
        for param, grad in zip(self.final_norm.parameters(), self.final_norm.gradients()):
            param -= learning_rate * grad

        # Update layers
        for layer in self.layers:
            for param, grad in zip(layer.parameters(), layer.gradients()):
                param -= learning_rate * grad

        # Update classifier
        self.classifier -= learning_rate * self.grad_classifier
        self.classifier_bias -= learning_rate * self.grad_classifier_bias

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        lr_decay: float = 0.95,
        lr_decay_epochs: int = 10,
        clip_grad: float = 1.0,
        verbose: bool = True
    ) -> Dict:
        """
        Train the transformer.

        Args:
            X_train: (num_samples, seq_len) token IDs
            y_train: (num_samples,) class labels
            X_val: Optional validation data
            y_val: Optional validation labels
            epochs: Number of training epochs
            batch_size: Mini-batch size
            learning_rate: Initial learning rate
            lr_decay: Learning rate decay factor
            lr_decay_epochs: Apply decay every N epochs
            clip_grad: Gradient clipping threshold
            verbose: Print progress

        Returns:
            Dictionary with training history
        """
        start_time = time.time()

        n_samples = len(X_train)
        n_batches = (n_samples + batch_size - 1) // batch_size

        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': [],
            'grad_norms': [],
        }

        current_lr = learning_rate

        if verbose:
            print("=" * 60)
            print("Standard Transformer Training")
            print("=" * 60)
            print(f"Samples: {n_samples}, Batches: {n_batches}, Epochs: {epochs}")
            print(f"Architecture: {self.config.num_layers} layers, {self.config.num_heads} heads")
            print(f"Parameters: {self.count_parameters():,}")
            print("=" * 60)

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0.0
            epoch_correct = 0

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)

                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # Forward pass
                self.zero_grad()
                logits = self.forward(X_batch, training=True)

                # Compute loss and gradient
                loss, grad = self._cross_entropy_loss(logits, y_batch)

                # Backward pass
                self.backward(grad)

                # Clip gradients
                grad_norm = self._clip_gradients(clip_grad)

                # Update parameters
                self._update_parameters(current_lr)

                # Track metrics
                epoch_loss += loss * len(y_batch)
                predictions = np.argmax(logits, axis=-1)
                epoch_correct += np.sum(predictions == y_batch)

            # Epoch metrics
            train_loss = epoch_loss / n_samples
            train_acc = epoch_correct / n_samples

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['learning_rates'].append(current_lr)
            history['grad_norms'].append(grad_norm)

            # Validation
            if X_val is not None and y_val is not None:
                val_logits = self.forward(X_val, training=False)
                val_loss, _ = self._cross_entropy_loss(val_logits, y_val)
                val_preds = np.argmax(val_logits, axis=-1)
                val_acc = np.mean(val_preds == y_val)

                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
            else:
                val_loss = None
                val_acc = None

            # Learning rate decay
            if (epoch + 1) % lr_decay_epochs == 0:
                current_lr *= lr_decay

            # Print progress
            if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                msg = f"Epoch {epoch:3d}: loss={train_loss:.4f}, acc={train_acc:.2%}"
                if val_loss is not None:
                    msg += f", val_loss={val_loss:.4f}, val_acc={val_acc:.2%}"
                print(msg)

        training_time = time.time() - start_time
        self.trained = True

        # Final metrics
        history['epochs_completed'] = epochs
        history['training_time_seconds'] = training_time
        history['final_train_loss'] = history['train_loss'][-1]
        history['final_train_acc'] = history['train_acc'][-1]
        if history['val_loss']:
            history['final_val_loss'] = history['val_loss'][-1]
            history['final_val_acc'] = history['val_acc'][-1]

        self.training_history = history

        if verbose:
            print("=" * 60)
            print(f"Training completed in {training_time:.2f}s")
            print(f"Final train accuracy: {train_acc:.2%}")
            if val_acc is not None:
                print(f"Final val accuracy: {val_acc:.2%}")
            print("=" * 60)

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Get predictions (class indices).

        Args:
            X: (num_samples, seq_len) token IDs

        Returns:
            (num_samples,) predicted class indices
        """
        logits = self.forward(X, training=False)
        return np.argmax(logits, axis=-1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            X: (num_samples, seq_len) token IDs

        Returns:
            (num_samples, num_classes) probability distribution
        """
        logits = self.forward(X, training=False)
        return self._softmax(logits)

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model on data.

        Args:
            X: (num_samples, seq_len) token IDs
            y: (num_samples,) true class labels

        Returns:
            Dictionary with metrics
        """
        logits = self.forward(X, training=False)
        loss, _ = self._cross_entropy_loss(logits, y)

        probs = self._softmax(logits)
        preds = np.argmax(probs, axis=-1)

        accuracy = np.mean(preds == y)

        # Top-3 accuracy
        top3_preds = np.argsort(probs, axis=-1)[:, -3:]
        top3_acc = np.mean([y[i] in top3_preds[i] for i in range(len(y))])

        # Top-5 accuracy
        top5_preds = np.argsort(probs, axis=-1)[:, -5:]
        top5_acc = np.mean([y[i] in top5_preds[i] for i in range(len(y))])

        # Perplexity
        perplexity = np.exp(loss)

        return {
            'loss': loss,
            'accuracy': accuracy,
            'top3_accuracy': top3_acc,
            'top5_accuracy': top5_acc,
            'perplexity': perplexity,
        }

    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        total = 0

        # Embedding
        total += sum(p.size for p in self.token_embedding.parameters())

        # Final norm
        total += sum(p.size for p in self.final_norm.parameters())

        # Encoder layers
        for layer in self.layers:
            total += layer.count_parameters()

        # Classifier
        total += self.classifier.size + self.classifier_bias.size

        return total

    def get_architecture_summary(self) -> Dict:
        """Get summary of model architecture"""
        return {
            'vocab_size': self.config.vocab_size,
            'embed_dim': self.config.embed_dim,
            'num_heads': self.config.num_heads,
            'num_layers': self.config.num_layers,
            'ffn_dim': self.config.ffn_dim,
            'num_classes': self.config.num_classes,
            'total_parameters': self.count_parameters(),
            'max_seq_len': self.config.max_seq_len,
        }

    def save(self, filepath: str) -> None:
        """Save model weights to file"""
        weights = {
            'config': {
                'vocab_size': self.config.vocab_size,
                'max_seq_len': self.config.max_seq_len,
                'embed_dim': self.config.embed_dim,
                'num_heads': self.config.num_heads,
                'num_layers': self.config.num_layers,
                'ffn_dim': self.config.ffn_dim,
                'dropout': self.config.dropout,
                'num_classes': self.config.num_classes,
                'seed': self.config.seed,
            },
            'token_embedding': self.token_embedding.weights,
            'classifier': self.classifier,
            'classifier_bias': self.classifier_bias,
            'final_norm_gamma': self.final_norm.gamma,
            'final_norm_beta': self.final_norm.beta,
            'training_history': self.training_history,
        }

        # Save layer weights
        for i, layer in enumerate(self.layers):
            weights[f'layer_{i}_attn_W_q'] = layer.attention.W_q
            weights[f'layer_{i}_attn_W_k'] = layer.attention.W_k
            weights[f'layer_{i}_attn_W_v'] = layer.attention.W_v
            weights[f'layer_{i}_attn_W_o'] = layer.attention.W_o
            weights[f'layer_{i}_attn_b_q'] = layer.attention.b_q
            weights[f'layer_{i}_attn_b_k'] = layer.attention.b_k
            weights[f'layer_{i}_attn_b_v'] = layer.attention.b_v
            weights[f'layer_{i}_attn_b_o'] = layer.attention.b_o
            weights[f'layer_{i}_ffn_W1'] = layer.ffn.W1
            weights[f'layer_{i}_ffn_b1'] = layer.ffn.b1
            weights[f'layer_{i}_ffn_W2'] = layer.ffn.W2
            weights[f'layer_{i}_ffn_b2'] = layer.ffn.b2
            weights[f'layer_{i}_norm1_gamma'] = layer.norm1.gamma
            weights[f'layer_{i}_norm1_beta'] = layer.norm1.beta
            weights[f'layer_{i}_norm2_gamma'] = layer.norm2.gamma
            weights[f'layer_{i}_norm2_beta'] = layer.norm2.beta

        np.savez(filepath, **weights)

    @classmethod
    def load(cls, filepath: str) -> "StandardTransformer":
        """Load model from file"""
        data = np.load(filepath, allow_pickle=True)

        # Reconstruct config
        config_dict = data['config'].item()
        config = TransformerConfig(**config_dict)

        # Create model
        model = cls(config)

        # Load weights
        model.token_embedding.weights = data['token_embedding']
        model.classifier = data['classifier']
        model.classifier_bias = data['classifier_bias']
        model.final_norm.gamma = data['final_norm_gamma']
        model.final_norm.beta = data['final_norm_beta']

        # Load training history if available
        if 'training_history' in data:
            model.training_history = data['training_history'].item()
            model.trained = True

        # Load layer weights
        for i, layer in enumerate(model.layers):
            layer.attention.W_q = data[f'layer_{i}_attn_W_q']
            layer.attention.W_k = data[f'layer_{i}_attn_W_k']
            layer.attention.W_v = data[f'layer_{i}_attn_W_v']
            layer.attention.W_o = data[f'layer_{i}_attn_W_o']
            layer.attention.b_q = data[f'layer_{i}_attn_b_q']
            layer.attention.b_k = data[f'layer_{i}_attn_b_k']
            layer.attention.b_v = data[f'layer_{i}_attn_b_v']
            layer.attention.b_o = data[f'layer_{i}_attn_b_o']
            layer.ffn.W1 = data[f'layer_{i}_ffn_W1']
            layer.ffn.b1 = data[f'layer_{i}_ffn_b1']
            layer.ffn.W2 = data[f'layer_{i}_ffn_W2']
            layer.ffn.b2 = data[f'layer_{i}_ffn_b2']
            layer.norm1.gamma = data[f'layer_{i}_norm1_gamma']
            layer.norm1.beta = data[f'layer_{i}_norm1_beta']
            layer.norm2.gamma = data[f'layer_{i}_norm2_gamma']
            layer.norm2.beta = data[f'layer_{i}_norm2_beta']

        return model
