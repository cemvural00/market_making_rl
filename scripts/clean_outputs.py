"""
Helper script to clean training outputs (models, results, or both).

Usage:
    # Delete all models
    python scripts/clean_outputs.py --models
    
    # Delete all results
    python scripts/clean_outputs.py --results
    
    # Delete both
    python scripts/clean_outputs.py --all
    
    # Delete specific agent/environment
    python scripts/clean_outputs.py --agent DeepPPOAgent --env ABMVanillaEnv
"""

import sys
import os
import argparse
import shutil

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def clean_models(agent_name=None, env_name=None):
    """Delete model files."""
    models_dir = "models"
    
    if not os.path.exists(models_dir):
        print("No models directory found.")
        return
    
    if agent_name and env_name:
        # Delete specific agent/env combination
        path = os.path.join(models_dir, env_name, agent_name)
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"✓ Deleted: {path}")
        else:
            print(f"✗ Not found: {path}")
    else:
        # Delete all models
        shutil.rmtree(models_dir)
        os.makedirs(models_dir, exist_ok=True)
        print("✓ Deleted all models")


def clean_results(agent_name=None, env_name=None):
    """Delete result files."""
    results_dir = "results"
    
    if not os.path.exists(results_dir):
        print("No results directory found.")
        return
    
    if agent_name and env_name:
        # Delete specific agent/env combination
        path = os.path.join(results_dir, env_name, agent_name)
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"✓ Deleted: {path}")
        else:
            print(f"✗ Not found: {path}")
    else:
        # Delete all results (but keep comparison tables if they exist)
        for item in os.listdir(results_dir):
            item_path = os.path.join(results_dir, item)
            if os.path.isdir(item_path) and item not in ["figures", "logs"]:
                shutil.rmtree(item_path)
                print(f"✓ Deleted: {item_path}")
        
        # Also delete comparison tables
        for file in os.listdir(results_dir):
            if file.startswith("comparison_"):
                file_path = os.path.join(results_dir, file)
                os.remove(file_path)
                print(f"✓ Deleted: {file_path}")
        
        print("✓ Deleted all results")


def main():
    parser = argparse.ArgumentParser(
        description="Clean training outputs (models, results, or both)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Delete all models
  python scripts/clean_outputs.py --models
  
  # Delete all results
  python scripts/clean_outputs.py --results
  
  # Delete both
  python scripts/clean_outputs.py --all
  
  # Delete specific agent/environment
  python scripts/clean_outputs.py --agent DeepPPOAgent --env ABMVanillaEnv --models
        """
    )
    
    parser.add_argument("--models", action="store_true",
                       help="Delete model files")
    parser.add_argument("--results", action="store_true",
                       help="Delete result files")
    parser.add_argument("--all", action="store_true",
                       help="Delete both models and results")
    parser.add_argument("--agent", type=str,
                       help="Specific agent name (requires --env)")
    parser.add_argument("--env", type=str,
                       help="Specific environment name (requires --agent)")
    
    args = parser.parse_args()
    
    # Validate specific deletion
    if (args.agent and not args.env) or (args.env and not args.agent):
        parser.error("--agent and --env must be used together")
    
    # Determine what to delete
    delete_models = args.models or args.all
    delete_results = args.results or args.all
    
    if not (delete_models or delete_results):
        parser.print_help()
        return
    
    # Confirm deletion
    if args.all:
        response = input("⚠️  This will delete ALL models and results. Continue? (yes/no): ")
    elif delete_models and delete_results:
        response = input("⚠️  This will delete models and results. Continue? (yes/no): ")
    elif delete_models:
        response = input("⚠️  This will delete all models. Continue? (yes/no): ")
    else:
        response = input("⚠️  This will delete all results. Continue? (yes/no): ")
    
    if response.lower() != "yes":
        print("Cancelled.")
        return
    
    # Perform deletion
    if delete_models:
        clean_models(args.agent, args.env)
    
    if delete_results:
        clean_results(args.agent, args.env)
    
    print("\n✓ Cleanup complete!")


if __name__ == "__main__":
    main()
