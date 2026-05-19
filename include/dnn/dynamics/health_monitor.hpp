#pragma once

#include "../core/network.hpp"
#include <deque>
#include <cmath>
#include <string>

namespace dnn {
namespace dynamics {

using core::Network;

/**
 * Health states for the network.
 */
enum class HealthState {
    Healthy,         // Normal operation
    CancerRisk,      // Too many layers being added
    Cancer,          // Pathological layer growth
    AlzheimerRisk,   // Too many layers being removed
    Alzheimer,       // Pathological layer loss
    Critical         // Network is non-functional
};

/**
 * Health report for the network.
 */
struct HealthReport {
    HealthState state = HealthState::Healthy;
    double cancer_score = 0.0;        // 0.0 - 1.0, higher = more cancer risk
    double alzheimer_score = 0.0;     // 0.0 - 1.0, higher = more alzheimer risk
    double overall_health = 1.0;      // 0.0 - 1.0, higher = healthier
    std::string diagnosis;
    std::vector<std::string> recommendations;
    size_t current_layers = 0;
    size_t current_nodes = 0;
    size_t changes_in_window = 0;
};

/**
 * Monitors network health for pathological growth patterns.
 *
 * Cancer: Excessive layer/node growth
 * - Network keeps adding capacity without improvement
 * - Wastes resources and slows training
 *
 * Alzheimer: Excessive layer/node removal
 * - Network keeps removing capacity, losing knowledge
 * - Can lead to underfitting and loss of learned patterns
 */
template<typename T = float>
class HealthMonitor {
public:
    explicit HealthMonitor(Network<T>& network,
                          double cancer_threshold = 0.7,
                          double alzheimer_threshold = 0.7,
                          size_t history_window = 50)
        : network_(network)
        , cancer_threshold_(cancer_threshold)
        , alzheimer_threshold_(alzheimer_threshold)
        , initial_layer_count_(network.num_layers())
        , initial_node_count_(network.num_nodes())
        , history_window_(history_window == 0 ? 50 : history_window) {}

    /**
     * Record a structural change event.
     */
    void record_change(int layer_delta, int node_delta) {
        ChangeEvent event;
        event.epoch = current_epoch_;
        event.layer_delta = layer_delta;
        event.node_delta = node_delta;

        change_history_.push_back(event);

        // Keep history limited
        while (change_history_.size() > history_window_) {
            change_history_.pop_front();
        }
    }

    /**
     * Update current epoch.
     */
    void update_epoch(uint64_t epoch) {
        current_epoch_ = epoch;
    }

    /**
     * Get comprehensive health diagnosis.
     */
    HealthReport diagnose() const {
        HealthReport report;

        report.current_layers = network_.num_layers();
        report.current_nodes = network_.num_nodes();
        report.changes_in_window = change_history_.size();

        report.cancer_score = compute_cancer_score();
        report.alzheimer_score = compute_alzheimer_score();

        // Determine state
        if (report.cancer_score >= 0.9) {
            report.state = HealthState::Cancer;
            report.diagnosis = "Pathological layer growth detected";
            report.recommendations.push_back("Stop adding layers immediately");
            report.recommendations.push_back("Consider pruning inefficient layers");
        } else if (report.cancer_score >= cancer_threshold_) {
            report.state = HealthState::CancerRisk;
            report.diagnosis = "Excessive layer growth pattern";
            report.recommendations.push_back("Reduce layer addition rate");
            report.recommendations.push_back("Evaluate layer efficiency before adding more");
        } else if (report.alzheimer_score >= 0.9) {
            report.state = HealthState::Alzheimer;
            report.diagnosis = "Pathological layer loss detected";
            report.recommendations.push_back("Stop removing layers immediately");
            report.recommendations.push_back("Review removal criteria");
        } else if (report.alzheimer_score >= alzheimer_threshold_) {
            report.state = HealthState::AlzheimerRisk;
            report.diagnosis = "Excessive layer removal pattern";
            report.recommendations.push_back("Reduce layer removal rate");
            report.recommendations.push_back("Evaluate if removed layers were truly inefficient");
        } else {
            report.state = HealthState::Healthy;
            report.diagnosis = "Network structure is stable";
        }

        // Check for critical state
        if (network_.num_layers() < 2) {
            report.state = HealthState::Critical;
            report.diagnosis = "Network has too few layers";
            report.recommendations.push_back("Minimum 2 layers required");
        }

        report.overall_health = 1.0 - std::max(report.cancer_score, report.alzheimer_score);

        return report;
    }

    /**
     * Get current health state.
     */
    HealthState current_state() const {
        return diagnose().state;
    }

    /**
     * Check if layer addition is allowed.
     */
    bool allow_layer_addition() const {
        auto report = diagnose();
        return report.state != HealthState::Cancer &&
               report.state != HealthState::CancerRisk &&
               report.state != HealthState::Critical;
    }

    /**
     * Check if layer removal is allowed.
     */
    bool allow_layer_removal() const {
        auto report = diagnose();
        return report.state != HealthState::Alzheimer &&
               report.state != HealthState::AlzheimerRisk &&
               report.state != HealthState::Critical &&
               network_.num_layers() > 2;  // Keep at least 2 layers
    }

    /**
     * Get cancer score.
     */
    double cancer_score() const {
        return compute_cancer_score();
    }

    /**
     * Get alzheimer score.
     */
    double alzheimer_score() const {
        return compute_alzheimer_score();
    }

    /**
     * Set thresholds.
     */
    void set_cancer_threshold(double threshold) {
        cancer_threshold_ = threshold;
    }

    void set_alzheimer_threshold(double threshold) {
        alzheimer_threshold_ = threshold;
    }

    /**
     * Reset history.
     */
    void reset() {
        change_history_.clear();
        initial_layer_count_ = network_.num_layers();
        initial_node_count_ = network_.num_nodes();
    }

private:
    struct ChangeEvent {
        uint64_t epoch = 0;
        int layer_delta = 0;
        int node_delta = 0;
    };

    double compute_cancer_score() const {
        if (change_history_.empty()) {
            return 0.0;
        }

        int total_layer_additions = 0;
        int total_node_additions = 0;
        int consecutive_additions = 0;
        int max_consecutive = 0;

        for (const auto& event : change_history_) {
            if (event.layer_delta > 0) {
                total_layer_additions += event.layer_delta;
                consecutive_additions++;
                max_consecutive = std::max(max_consecutive, consecutive_additions);
            } else if (event.layer_delta < 0) {
                consecutive_additions = 0;
            }

            if (event.node_delta > 0) {
                total_node_additions += event.node_delta;
            }
        }

        // Factors:
        // 1. Rate of layer addition
        double layer_rate = static_cast<double>(total_layer_additions) / change_history_.size();

        // 2. Consecutive additions
        double consecutive_factor = static_cast<double>(max_consecutive) / change_history_.size();

        // 3. Size growth ratio
        double size_ratio = static_cast<double>(network_.num_layers()) / initial_layer_count_;
        double size_factor = std::max(0.0, (size_ratio - 1.0) / 5.0);  // Normalize

        // 4. Node growth ratio
        double node_ratio = static_cast<double>(network_.num_nodes()) / std::max(size_t(1), initial_node_count_);
        double node_factor = std::max(0.0, (node_ratio - 1.0) / 10.0);

        // Combine
        double score =
            0.3 * layer_rate +
            0.25 * consecutive_factor +
            0.25 * size_factor +
            0.2 * node_factor;

        return std::min(1.0, score);
    }

    double compute_alzheimer_score() const {
        if (change_history_.empty()) {
            return 0.0;
        }

        int total_layer_removals = 0;
        int total_node_removals = 0;
        int consecutive_removals = 0;
        int max_consecutive = 0;

        for (const auto& event : change_history_) {
            if (event.layer_delta < 0) {
                total_layer_removals += -event.layer_delta;
                consecutive_removals++;
                max_consecutive = std::max(max_consecutive, consecutive_removals);
            } else if (event.layer_delta > 0) {
                consecutive_removals = 0;
            }

            if (event.node_delta < 0) {
                total_node_removals += -event.node_delta;
            }
        }

        // Factors:
        // 1. Rate of layer removal
        double layer_rate = static_cast<double>(total_layer_removals) / change_history_.size();

        // 2. Consecutive removals
        double consecutive_factor = static_cast<double>(max_consecutive) / change_history_.size();

        // 3. Size shrink ratio
        double size_ratio = static_cast<double>(initial_layer_count_) /
                           std::max(size_t(1), network_.num_layers());
        double size_factor = std::max(0.0, (size_ratio - 1.0) / 3.0);

        // 4. Approaching minimum
        double min_factor = (network_.num_layers() <= 2) ? 1.0 : 0.0;

        // Combine
        double score =
            0.3 * layer_rate +
            0.25 * consecutive_factor +
            0.25 * size_factor +
            0.2 * min_factor;

        return std::min(1.0, score);
    }

    Network<T>& network_;
    double cancer_threshold_;
    double alzheimer_threshold_;
    size_t initial_layer_count_;
    size_t initial_node_count_;
    size_t history_window_;
    uint64_t current_epoch_ = 0;
    std::deque<ChangeEvent> change_history_;
};

/**
 * Get string name for health state.
 */
inline std::string health_state_name(HealthState state) {
    switch (state) {
        case HealthState::Healthy: return "Healthy";
        case HealthState::CancerRisk: return "Cancer Risk";
        case HealthState::Cancer: return "Cancer";
        case HealthState::AlzheimerRisk: return "Alzheimer Risk";
        case HealthState::Alzheimer: return "Alzheimer";
        case HealthState::Critical: return "Critical";
        default: return "Unknown";
    }
}

} // namespace dynamics
} // namespace dnn
