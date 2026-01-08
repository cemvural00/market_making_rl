"""
Main entry point for the full training and comparison pipeline.

This script can:
1. Train all RL agents on all environments
2. Compare all agents (RL + heuristic) on all environments
3. Aggregate results into comparison tables
4. Run the full pipeline (train -> compare -> aggregate)
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_all_rl_agents import train_all
from scripts.compare_all_agents import compare_all
from scripts.aggregate_results import main as aggregate_results


def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline for training and comparing agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all RL agents
  python scripts/run_full_pipeline.py --train
  
  # Compare all agents (assumes models exist)
  python scripts/run_full_pipeline.py --compare
  
  # Full pipeline: train + compare + aggregate
  python scripts/run_full_pipeline.py --full
  
  # Train with custom evaluation episodes
  python scripts/run_full_pipeline.py --train --train-eval-episodes 200
        """
    )
    
    parser.add_argument("--train", action="store_true",
                       help="Train all RL agents on all environments")
    parser.add_argument("--compare", action="store_true",
                       help="Compare all agents on all environments")
    parser.add_argument("--aggregate", action="store_true",
                       help="Aggregate results into comparison tables")
    parser.add_argument("--full", action="store_true",
                       help="Run full pipeline: train -> compare -> aggregate")
    
    # Training options
    parser.add_argument("--train-eval-episodes", type=int, default=100,
                       help="Evaluation episodes after training (default: 100)")
    parser.add_argument("--no-skip-train", action="store_true",
                       help="Retrain even if model exists")
    
    # Comparison options
    parser.add_argument("--compare-eval-episodes", type=int, default=100,
                       help="Evaluation episodes for comparison (default: 100)")
    parser.add_argument("--no-skip-compare", action="store_true",
                       help="Re-run comparison even if results exist")
    parser.add_argument("--rl-only", action="store_true",
                       help="Only compare RL agents")
    parser.add_argument("--heuristic-only", action="store_true",
                       help="Only compare heuristic agents")
    
    args = parser.parse_args()
    
    # If --full is specified, run everything
    if args.full:
        args.train = True
        args.compare = True
        args.aggregate = True
    
    # If no action specified, show help
    if not (args.train or args.compare or args.aggregate):
        parser.print_help()
        return
    
    # Training phase
    if args.train:
        print("\n" + "="*60)
        print("PHASE 1: TRAINING RL AGENTS")
        print("="*60)
        train_all(
            skip_if_exists=not args.no_skip_train,
            n_eval_episodes=args.train_eval_episodes
        )
    
    # Comparison phase
    if args.compare:
        print("\n" + "="*60)
        print("PHASE 2: COMPARING ALL AGENTS")
        print("="*60)
        compare_all(
            n_eval_episodes=args.compare_eval_episodes,
            skip_if_exists=not args.no_skip_compare,
            rl_only=args.rl_only,
            heuristic_only=args.heuristic_only
        )
    
    # Aggregation phase
    if args.aggregate:
        print("\n" + "="*60)
        print("PHASE 3: AGGREGATING RESULTS")
        print("="*60)
        aggregate_results()
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print("\nResults saved to:")
    print("  - Models: models/")
    print("  - Results: results/")
    print("  - Comparison tables: results/comparison_*.csv/json/md")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
