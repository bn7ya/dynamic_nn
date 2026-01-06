"""
Main entry point for Sentiment Analysis project.
Arabic + English Bilingual Sentiment Classification using Dynamic Neural Network.

Usage:
    python main.py                    # Run with default settings (120k samples)
    python main.py --samples 50000    # Generate 50k samples
    python main.py --skip-generate    # Skip data generation, use existing data
"""

import sys
import os
import argparse
import time

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

from src.utils import setup_unicode_output, print_header, Timer
from src.data_generator import SentimentDataGenerator, generate_dataset
from src.tokenizer import SimpleTokenizer, normalize_features, one_hot_encode
from src.trainer import SentimentTrainer
from src.evaluator import SentimentEvaluator
import config

# Setup Unicode for Windows
setup_unicode_output()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Sentiment Analysis with Dynamic Neural Network"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=config.DEFAULT_NUM_SAMPLES,
        help=f"Number of samples to generate (default: {config.DEFAULT_NUM_SAMPLES})"
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip data generation, use existing CSV files"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.RANDOM_SEED,
        help=f"Random seed (default: {config.RANDOM_SEED})"
    )
    parser.add_argument(
        "--max-vocab",
        type=int,
        default=config.MAX_VOCAB_SIZE,
        help=f"Maximum vocabulary size (default: {config.MAX_VOCAB_SIZE})"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save model and plots"
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    print_header("SENTIMENT ANALYSIS WITH DYNAMIC NEURAL NETWORK")
    print("Arabic + English Bilingual Classification")
    print("=" * 60)

    total_start = time.time()
    timings = {}

    # =========================================================================
    # Step 1: Data Generation / Loading
    # =========================================================================
    print_header("STEP 1: DATA PREPARATION", char="-")

    generator = SentimentDataGenerator(seed=args.seed)

    if args.skip_generate and os.path.exists(config.TRAIN_FILE):
        print("Loading existing data files...")
        with Timer("Data loading") as t:
            train_texts, train_labels, _, _ = generator.load_from_csv(config.TRAIN_FILE)
            test_texts, test_labels, _, _ = generator.load_from_csv(config.TEST_FILE)
        timings['data_loading'] = t.elapsed
    else:
        print(f"Generating {args.samples:,} samples...")
        with Timer("Data generation") as t:
            generate_dataset(
                args.samples,
                config.DATA_DIR,
                train_ratio=config.TRAIN_RATIO,
                seed=args.seed
            )
            train_texts, train_labels, _, _ = generator.load_from_csv(config.TRAIN_FILE)
            test_texts, test_labels, _, _ = generator.load_from_csv(config.TEST_FILE)
        timings['data_generation'] = t.elapsed

    print(f"\nDataset loaded:")
    print(f"  Training samples: {len(train_texts):,}")
    print(f"  Test samples: {len(test_texts):,}")

    # =========================================================================
    # Step 2: Data Exploration
    # =========================================================================
    print_header("STEP 2: DATA EXPLORATION", char="-")

    # Distribution
    print("\nTraining set distribution:")
    for label_id, label_name in config.LABEL_NAMES.items():
        count = train_labels.count(label_id)
        pct = count / len(train_labels) * 100
        print(f"  {label_name}: {count:,} ({pct:.1f}%)")

    # Word statistics
    import numpy as np
    word_counts = [len(text.split()) for text in train_texts]
    print(f"\nWord count statistics:")
    print(f"  Min: {min(word_counts)}, Max: {max(word_counts)}, Avg: {np.mean(word_counts):.2f}")

    # =========================================================================
    # Step 3: Tokenization
    # =========================================================================
    print_header("STEP 3: TOKENIZATION", char="-")

    with Timer("Tokenization") as t:
        tokenizer = SimpleTokenizer(max_vocab_size=args.max_vocab)

        # Fit on training data
        tokenizer.fit(train_texts)

        # Transform both sets
        print("\nTransforming training data...")
        X_train = tokenizer.transform_bow(train_texts)

        print("\nTransforming test data...")
        X_test = tokenizer.transform_bow(test_texts)

    timings['tokenization'] = t.elapsed

    # =========================================================================
    # Step 4: Normalization
    # =========================================================================
    print_header("STEP 4: NORMALIZATION", char="-")

    X_train = normalize_features(X_train)
    X_test = normalize_features(X_test)

    y_train = one_hot_encode(train_labels, config.NUM_CLASSES)
    y_test = one_hot_encode(test_labels, config.NUM_CLASSES)

    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    print(f"Training labels shape: {y_train.shape}")

    # =========================================================================
    # Step 5: Training
    # =========================================================================
    print_header("STEP 5: TRAINING", char="-")

    trainer = SentimentTrainer(
        input_size=X_train.shape[1],
        num_classes=config.NUM_CLASSES,
        seed=config.MODEL_SEED,
        cost_function=config.COST_FUNCTION
    )

    with Timer("Training") as t:
        metrics = trainer.train(X_train, y_train, verbose=True)

    timings['training'] = t.elapsed

    # =========================================================================
    # Step 6: Evaluation
    # =========================================================================
    print_header("STEP 6: EVALUATION", char="-")

    evaluator = SentimentEvaluator(label_names=config.LABEL_NAMES)

    with Timer("Evaluation") as t:
        eval_metrics = evaluator.evaluate_from_trainer(trainer, X_test, y_test)

    timings['evaluation'] = t.elapsed

    # Show examples
    predictions = trainer.predict(X_test)
    evaluator.show_prediction_examples(test_texts, predictions, y_test, n=5)

    # Network health
    print("\nNetwork Health:")
    health = trainer.get_health_status()
    print(f"  State: {health['state']}")
    print(f"  Diagnosis: {health['diagnosis']}")
    print(f"  Overall health: {health['overall_health']:.4f}")

    # =========================================================================
    # Step 7: Save Model and Visualizations
    # =========================================================================
    if not args.no_save:
        print_header("STEP 7: SAVING", char="-")

        try:
            os.makedirs(config.MODELS_DIR, exist_ok=True)
            trainer.save(config.MODELS_DIR, config.MODEL_NAME)
        except Exception as e:
            print(f"Model save skipped: {e}")

        try:
            from pydnn import auto_generate_plots
            os.makedirs(config.PLOTS_DIR, exist_ok=True)
            result = trainer.get_training_result()
            saved_files = auto_generate_plots(
                result,
                network=trainer.network,
                output_dir=config.PLOTS_DIR,
                prefix="sentiment",
                show=False
            )
            print(f"Plots saved: {saved_files}")
        except Exception as e:
            print(f"Visualization skipped: {e}")

    # =========================================================================
    # Summary
    # =========================================================================
    total_time = time.time() - total_start

    print_header("PERFORMANCE SUMMARY")

    print(f"\nDataset:")
    print(f"  Training samples: {len(train_texts):,}")
    print(f"  Test samples: {len(test_texts):,}")
    print(f"  Vocabulary size: {tokenizer.vocab_size:,}")
    print(f"  Feature dimensions: {X_train.shape[1]:,}")

    print(f"\nTiming:")
    for name, elapsed in timings.items():
        print(f"  {name.replace('_', ' ').title()}: {elapsed:.2f}s")
    print(f"  Total time: {total_time:.2f}s")

    print(f"\nThroughput:")
    print(f"  Training: {metrics.samples_per_second:,.0f} samples/sec")
    print(f"  Prediction: {eval_metrics.predictions_per_second:,.0f} samples/sec")

    print(f"\nResults:")
    print(f"  Test Accuracy: {eval_metrics.accuracy:.4f} ({eval_metrics.accuracy*100:.2f}%)")
    print(f"  Final Cost: {metrics.final_cost:.6f}")

    print_header("SENTIMENT ANALYSIS COMPLETE")

    return trainer, tokenizer, eval_metrics


if __name__ == "__main__":
    trainer, tokenizer, metrics = main()
