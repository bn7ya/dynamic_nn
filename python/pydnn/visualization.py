"""
Visualization tools for Dynamic Neural Network training.
Generates training history plots, network structure diagrams, and health timelines.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.collections import PatchCollection
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


@dataclass
class TrainingVisualization:
    """Container for training visualization data."""
    cost_history: List[float]
    efficiency_history: List[float]
    layer_history: Optional[List[int]] = None
    node_history: Optional[List[int]] = None
    health_history: Optional[List[Dict[str, float]]] = None
    batch_size_history: Optional[List[int]] = None
    learning_rate_history: Optional[List[float]] = None
    # Emotional state data
    depression_history: Optional[List[float]] = None
    excitement_history: Optional[List[float]] = None
    emotional_actions: Optional[List[str]] = None


class TrainingPlotter:
    """
    Creates visualizations for training progress.
    Supports both matplotlib (static) and plotly (interactive).
    """

    def __init__(self, use_plotly: bool = False):
        """
        Initialize the plotter.

        Args:
            use_plotly: Use plotly for interactive plots (default: matplotlib)
        """
        self.use_plotly = use_plotly and HAS_PLOTLY

        if not HAS_MATPLOTLIB and not HAS_PLOTLY:
            raise ImportError(
                "No visualization library available. "
                "Install matplotlib or plotly: pip install matplotlib plotly"
            )

    def plot_training_history(self,
                              data: TrainingVisualization,
                              save_path: Optional[str] = None,
                              show: bool = True,
                              title: str = "Training History") -> Any:
        """
        Plot comprehensive training history.

        Args:
            data: Training visualization data
            save_path: Path to save the figure (optional)
            show: Display the plot
            title: Plot title

        Returns:
            Figure object (matplotlib or plotly)
        """
        if self.use_plotly:
            return self._plot_training_plotly(data, save_path, show, title)
        else:
            return self._plot_training_matplotlib(data, save_path, show, title)

    def _plot_training_matplotlib(self, data: TrainingVisualization,
                                  save_path: Optional[str],
                                  show: bool, title: str):
        """Create matplotlib training plots."""
        n_plots = 2
        if data.layer_history:
            n_plots += 1
        if data.health_history:
            n_plots += 1
        if data.depression_history and data.excitement_history:
            n_plots += 1

        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
        if n_plots == 1:
            axes = [axes]

        epochs = list(range(len(data.cost_history)))
        plot_idx = 0

        # Cost history
        ax = axes[plot_idx]
        ax.plot(epochs, data.cost_history, 'b-', linewidth=2, label='Training Cost')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cost')
        ax.set_title('Cost Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if data.learning_rate_history:
            ax2 = ax.twinx()
            ax2.plot(epochs[:len(data.learning_rate_history)],
                    data.learning_rate_history, 'g--', alpha=0.5, label='Learning Rate')
            ax2.set_ylabel('Learning Rate', color='g')
            ax2.tick_params(axis='y', labelcolor='g')
        plot_idx += 1

        # Efficiency history
        ax = axes[plot_idx]
        ax.plot(epochs, data.efficiency_history, 'g-', linewidth=2, label='Network Efficiency')
        ax.fill_between(epochs, data.efficiency_history, alpha=0.3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Efficiency')
        ax.set_title('Efficiency Over Time')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

        # Layer/node history
        if data.layer_history:
            ax = axes[plot_idx]
            ax.plot(epochs[:len(data.layer_history)], data.layer_history,
                   'r-', linewidth=2, label='Layers')
            if data.node_history:
                ax2 = ax.twinx()
                ax2.plot(epochs[:len(data.node_history)], data.node_history,
                        'orange', linewidth=2, label='Nodes')
                ax2.set_ylabel('Total Nodes', color='orange')
                ax2.tick_params(axis='y', labelcolor='orange')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Number of Layers', color='r')
            ax.tick_params(axis='y', labelcolor='r')
            ax.set_title('Network Architecture Evolution')
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        # Health history
        if data.health_history:
            ax = axes[plot_idx]
            cancer_scores = [h.get('cancer_score', 0) for h in data.health_history]
            alzheimer_scores = [h.get('alzheimer_score', 0) for h in data.health_history]
            health_epochs = range(len(data.health_history))

            ax.plot(health_epochs, cancer_scores, 'r-', linewidth=2, label='Cancer Score')
            ax.plot(health_epochs, alzheimer_scores, 'b-', linewidth=2, label='Alzheimer Score')
            ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='Risk Threshold')
            ax.fill_between(health_epochs, cancer_scores, alpha=0.2, color='r')
            ax.fill_between(health_epochs, alzheimer_scores, alpha=0.2, color='b')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Score')
            ax.set_title('Network Health Over Time')
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        # Emotional state history (depression/excitement)
        if data.depression_history and data.excitement_history:
            ax = axes[plot_idx]
            emotional_epochs = range(len(data.depression_history))

            ax.plot(emotional_epochs, data.depression_history, 'b-', linewidth=2, label='Depression Ratio')
            ax.plot(emotional_epochs, data.excitement_history, 'g-', linewidth=2, label='Excitement Ratio')
            ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Extreme Threshold')
            ax.fill_between(emotional_epochs, data.depression_history, alpha=0.2, color='b')
            ax.fill_between(emotional_epochs, data.excitement_history, alpha=0.2, color='g')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Ratio')
            ax.set_title('Emotional State Over Time (Depression/Excitement)')
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    def _plot_training_plotly(self, data: TrainingVisualization,
                              save_path: Optional[str],
                              show: bool, title: str):
        """Create plotly interactive training plots."""
        n_rows = 2
        if data.layer_history:
            n_rows += 1
        if data.health_history:
            n_rows += 1
        if data.depression_history and data.excitement_history:
            n_rows += 1

        specs = [[{"secondary_y": True}] for _ in range(n_rows)]
        subplot_titles = ['Cost Over Time', 'Efficiency Over Time']
        if data.layer_history:
            subplot_titles.append('Network Architecture Evolution')
        if data.health_history:
            subplot_titles.append('Network Health Over Time')
        if data.depression_history and data.excitement_history:
            subplot_titles.append('Emotional State Over Time')

        fig = make_subplots(rows=n_rows, cols=1,
                           specs=specs,
                           subplot_titles=subplot_titles,
                           vertical_spacing=0.08)

        epochs = list(range(len(data.cost_history)))
        row = 1

        # Cost history
        fig.add_trace(
            go.Scatter(x=epochs, y=data.cost_history, mode='lines',
                      name='Training Cost', line=dict(color='blue', width=2)),
            row=row, col=1
        )

        if data.learning_rate_history:
            fig.add_trace(
                go.Scatter(x=epochs[:len(data.learning_rate_history)],
                          y=data.learning_rate_history, mode='lines',
                          name='Learning Rate', line=dict(color='green', width=1, dash='dash')),
                row=row, col=1, secondary_y=True
            )
        row += 1

        # Efficiency history
        fig.add_trace(
            go.Scatter(x=epochs, y=data.efficiency_history, mode='lines',
                      name='Network Efficiency', line=dict(color='green', width=2),
                      fill='tozeroy', fillcolor='rgba(0,255,0,0.1)'),
            row=row, col=1
        )
        row += 1

        # Layer/node history
        if data.layer_history:
            fig.add_trace(
                go.Scatter(x=epochs[:len(data.layer_history)], y=data.layer_history,
                          mode='lines', name='Layers', line=dict(color='red', width=2)),
                row=row, col=1
            )
            if data.node_history:
                fig.add_trace(
                    go.Scatter(x=epochs[:len(data.node_history)], y=data.node_history,
                              mode='lines', name='Nodes', line=dict(color='orange', width=2)),
                    row=row, col=1, secondary_y=True
                )
            row += 1

        # Health history
        if data.health_history:
            cancer_scores = [h.get('cancer_score', 0) for h in data.health_history]
            alzheimer_scores = [h.get('alzheimer_score', 0) for h in data.health_history]
            health_epochs = list(range(len(data.health_history)))

            fig.add_trace(
                go.Scatter(x=health_epochs, y=cancer_scores, mode='lines',
                          name='Cancer Score', line=dict(color='red', width=2),
                          fill='tozeroy', fillcolor='rgba(255,0,0,0.1)'),
                row=row, col=1
            )
            fig.add_trace(
                go.Scatter(x=health_epochs, y=alzheimer_scores, mode='lines',
                          name='Alzheimer Score', line=dict(color='blue', width=2),
                          fill='tozeroy', fillcolor='rgba(0,0,255,0.1)'),
                row=row, col=1
            )
            fig.add_hline(y=0.7, line_dash="dash", line_color="orange",
                         annotation_text="Risk Threshold", row=row, col=1)
            row += 1

        # Emotional state history (depression/excitement)
        if data.depression_history and data.excitement_history:
            emotional_epochs = list(range(len(data.depression_history)))

            fig.add_trace(
                go.Scatter(x=emotional_epochs, y=data.depression_history, mode='lines',
                          name='Depression Ratio', line=dict(color='blue', width=2),
                          fill='tozeroy', fillcolor='rgba(0,0,255,0.1)'),
                row=row, col=1
            )
            fig.add_trace(
                go.Scatter(x=emotional_epochs, y=data.excitement_history, mode='lines',
                          name='Excitement Ratio', line=dict(color='green', width=2),
                          fill='tozeroy', fillcolor='rgba(0,255,0,0.1)'),
                row=row, col=1
            )
            fig.add_hline(y=0.8, line_dash="dash", line_color="red",
                         annotation_text="Extreme Threshold", row=row, col=1)

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            height=300 * n_rows,
            showlegend=True,
            template='plotly_white'
        )

        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path)

        if show:
            fig.show()

        return fig


class NetworkVisualizer:
    """
    Visualizes network architecture.
    Shows layers, nodes, and connections.
    """

    def __init__(self):
        if not HAS_MATPLOTLIB:
            raise ImportError("NetworkVisualizer requires matplotlib")

    def draw_network(self,
                     layer_sizes: List[int],
                     layer_efficiencies: Optional[List[float]] = None,
                     node_efficiencies: Optional[List[List[float]]] = None,
                     save_path: Optional[str] = None,
                     show: bool = True,
                     title: str = "Network Architecture") -> Any:
        """
        Draw network architecture diagram.

        Args:
            layer_sizes: Number of nodes in each layer
            layer_efficiencies: Efficiency score for each layer (0-1)
            node_efficiencies: Efficiency score for each node (optional)
            save_path: Path to save the figure
            show: Display the plot
            title: Plot title

        Returns:
            matplotlib Figure
        """
        n_layers = len(layer_sizes)
        max_nodes = max(layer_sizes)

        fig, ax = plt.subplots(figsize=(max(12, n_layers * 2), max(8, max_nodes * 0.3)))

        # Layout parameters
        layer_spacing = 1.0 / (n_layers + 1)
        node_radius = min(0.02, 0.3 / max_nodes)

        # Color map for efficiency
        cmap = plt.cm.RdYlGn

        for layer_idx, n_nodes in enumerate(layer_sizes):
            x = (layer_idx + 1) * layer_spacing

            # Compute vertical positions
            if n_nodes == 1:
                y_positions = [0.5]
            else:
                y_positions = np.linspace(0.1, 0.9, n_nodes)

            # Get layer efficiency color
            if layer_efficiencies and layer_idx < len(layer_efficiencies):
                layer_color = cmap(layer_efficiencies[layer_idx])
            else:
                layer_color = 'lightblue'

            # Draw nodes
            for node_idx, y in enumerate(y_positions):
                # Get node color
                if node_efficiencies and layer_idx < len(node_efficiencies):
                    if node_idx < len(node_efficiencies[layer_idx]):
                        color = cmap(node_efficiencies[layer_idx][node_idx])
                    else:
                        color = layer_color
                else:
                    color = layer_color

                circle = plt.Circle((x, y), node_radius, color=color,
                                    ec='black', linewidth=0.5, zorder=2)
                ax.add_patch(circle)

            # Draw connections to next layer (simplified for large networks)
            if layer_idx < n_layers - 1:
                next_n_nodes = layer_sizes[layer_idx + 1]
                next_x = (layer_idx + 2) * layer_spacing

                if n_nodes == 1:
                    next_y_positions = [0.5]
                else:
                    next_y_positions = np.linspace(0.1, 0.9, next_n_nodes)

                # Limit connections for visualization (sample if too many)
                max_connections = 100
                if n_nodes * next_n_nodes > max_connections:
                    # Draw representative connections
                    sample_from = np.linspace(0, n_nodes - 1, min(5, n_nodes)).astype(int)
                    sample_to = np.linspace(0, next_n_nodes - 1, min(5, next_n_nodes)).astype(int)
                else:
                    sample_from = range(n_nodes)
                    sample_to = range(next_n_nodes)

                for i in sample_from:
                    for j in sample_to:
                        ax.plot([x, next_x],
                               [y_positions[i], next_y_positions[j]],
                               'gray', alpha=0.1, linewidth=0.3, zorder=1)

            # Add layer label
            ax.text(x, -0.05, f'Layer {layer_idx}\n({n_nodes} nodes)',
                   ha='center', va='top', fontsize=9)

        # Add colorbar for efficiency
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
        cbar.set_label('Efficiency Score')

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.15, 1.05)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()

        return fig


class HealthTimeline:
    """
    Visualizes network health over time with state changes.
    """

    def __init__(self):
        if not HAS_MATPLOTLIB:
            raise ImportError("HealthTimeline requires matplotlib")

    def draw_timeline(self,
                      health_history: List[Dict[str, Any]],
                      save_path: Optional[str] = None,
                      show: bool = True,
                      title: str = "Network Health Timeline") -> Any:
        """
        Draw health timeline showing state transitions.

        Args:
            health_history: List of health reports over time
            save_path: Path to save the figure
            show: Display the plot
            title: Plot title

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        epochs = range(len(health_history))

        # State colors
        state_colors = {
            'Healthy': 'green',
            'CancerRisk': 'yellow',
            'Cancer': 'red',
            'AlzheimerRisk': 'cyan',
            'Alzheimer': 'blue',
            'Critical': 'purple'
        }

        # Plot 1: Cancer and Alzheimer scores
        ax = axes[0]
        cancer_scores = [h.get('cancer_score', 0) for h in health_history]
        alzheimer_scores = [h.get('alzheimer_score', 0) for h in health_history]

        ax.fill_between(epochs, cancer_scores, alpha=0.3, color='red', label='Cancer Score')
        ax.fill_between(epochs, alzheimer_scores, alpha=0.3, color='blue', label='Alzheimer Score')
        ax.plot(epochs, cancer_scores, 'r-', linewidth=2)
        ax.plot(epochs, alzheimer_scores, 'b-', linewidth=2)
        ax.axhline(y=0.7, color='orange', linestyle='--', linewidth=1.5, label='Risk Threshold')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right')
        ax.set_title('Health Scores')
        ax.grid(True, alpha=0.3)

        # Plot 2: Overall health
        ax = axes[1]
        overall_health = [h.get('overall_health', 1) for h in health_history]
        colors = ['green' if h > 0.7 else 'yellow' if h > 0.4 else 'red' for h in overall_health]

        ax.bar(epochs, overall_health, color=colors, alpha=0.7, width=1.0)
        ax.axhline(y=0.7, color='green', linestyle='--', linewidth=1, alpha=0.7)
        ax.axhline(y=0.4, color='orange', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_ylabel('Overall Health')
        ax.set_ylim(0, 1)
        ax.set_title('Overall Network Health')
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 3: State timeline
        ax = axes[2]
        states = [h.get('state', 'Healthy') for h in health_history]

        # Create state regions
        current_state = states[0]
        start_epoch = 0

        patches = []
        for i, state in enumerate(states):
            if state != current_state or i == len(states) - 1:
                end_epoch = i if state != current_state else i + 1
                color = state_colors.get(current_state, 'gray')
                rect = mpatches.Rectangle((start_epoch, 0), end_epoch - start_epoch, 1,
                                          facecolor=color, alpha=0.6)
                patches.append(rect)
                ax.text((start_epoch + end_epoch) / 2, 0.5, current_state,
                       ha='center', va='center', fontsize=9, rotation=90 if (end_epoch - start_epoch) < 10 else 0)
                current_state = state
                start_epoch = i

        for patch in patches:
            ax.add_patch(patch)

        ax.set_xlim(0, len(states))
        ax.set_ylim(0, 1)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('State')
        ax.set_title('Health State Timeline')
        ax.set_yticks([])

        # Add legend for states
        legend_patches = [mpatches.Patch(color=color, label=state, alpha=0.6)
                         for state, color in state_colors.items()]
        ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                 ncol=6, fontsize=8)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()

        return fig


def auto_generate_plots(training_result,
                        network=None,
                        health_history: Optional[List[Dict]] = None,
                        output_dir: str = ".",
                        prefix: str = "training",
                        show: bool = False,
                        use_plotly: bool = False) -> List[str]:
    """
    Automatically generate all relevant plots after training.

    Args:
        training_result: TrainingResult from training
        network: DynamicNetwork instance (optional, for architecture plot)
        health_history: List of health reports (optional)
        output_dir: Directory to save plots
        prefix: Prefix for plot filenames
        show: Show plots interactively
        use_plotly: Use plotly for interactive plots

    Returns:
        List of saved file paths
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    saved_files = []

    # Training history plot
    plotter = TrainingPlotter(use_plotly=use_plotly)

    # Get emotional state data if available
    depression_history = getattr(training_result, 'depression_history', None)
    excitement_history = getattr(training_result, 'excitement_history', None)
    learning_rate_history = getattr(training_result, 'learning_rate_history', None)

    data = TrainingVisualization(
        cost_history=training_result.cost_history,
        efficiency_history=training_result.efficiency_history,
        health_history=health_history,
        learning_rate_history=learning_rate_history,
        depression_history=depression_history,
        excitement_history=excitement_history
    )

    ext = '.html' if use_plotly else '.png'
    history_path = os.path.join(output_dir, f"{prefix}_history{ext}")
    plotter.plot_training_history(data, save_path=history_path, show=show)
    saved_files.append(history_path)

    # Network architecture plot (matplotlib only)
    if network is not None and HAS_MATPLOTLIB:
        try:
            viz = NetworkVisualizer()

            # Get layer sizes from network
            if hasattr(network, 'num_layers') and hasattr(network, '_layers'):
                layer_sizes = []
                layer_effs = []
                for layer in network._layers:
                    layer_sizes.append(layer.get('W', np.array([[]])).shape[0])
                    layer_effs.append(0.5)  # Placeholder

                if layer_sizes:
                    arch_path = os.path.join(output_dir, f"{prefix}_architecture.png")
                    viz.draw_network(layer_sizes, layer_efficiencies=layer_effs,
                                    save_path=arch_path, show=show)
                    saved_files.append(arch_path)
        except Exception as e:
            print(f"Could not generate architecture plot: {e}")

    # Health timeline
    if health_history and HAS_MATPLOTLIB:
        try:
            timeline = HealthTimeline()
            health_path = os.path.join(output_dir, f"{prefix}_health.png")
            timeline.draw_timeline(health_history, save_path=health_path, show=show)
            saved_files.append(health_path)
        except Exception as e:
            print(f"Could not generate health timeline: {e}")

    return saved_files
