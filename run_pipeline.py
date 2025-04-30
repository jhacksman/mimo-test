"""Run the complete pipeline for fine-tuning MiMo-7B for tool-calling."""

import os
import argparse
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and print its output."""
    print(f"\n=== {description} ===\n")
    result = subprocess.run(command, shell=True, check=True)
    return result.returncode == 0

def main():
    """Run the complete pipeline."""
    parser = argparse.ArgumentParser(description="Run the MiMo-7B tool-calling fine-tuning pipeline")
    parser.add_argument("--skip-data-gen", action="store_true", help="Skip data generation step")
    parser.add_argument("--skip-validation", action="store_true", help="Skip data validation step")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training step")
    parser.add_argument("--test-only", action="store_true", help="Only run the model testing step")
    args = parser.parse_args()
    
    if not os.environ.get("OPENROUTER_API_KEY") and not args.skip_data_gen:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Please set it using: export OPENROUTER_API_KEY=your_key_here")
        return False
    
    success = True
    
    if args.test_only:
        success = run_command("python -m src.fine_tuning.test_model", "Testing fine-tuned model")
    else:
        if not args.skip_data_gen:
            success = run_command("python -m src.data_generation.generate_data", "Generating training data")
            if not success:
                return False
        
        if not args.skip_validation:
            success = run_command("python -m src.data_validation.validate_data", "Validating training data")
            if not success:
                return False
        
        if not args.skip_training:
            success = run_command("python -m src.fine_tuning.train", "Fine-tuning the model")
            if not success:
                return False
        
        success = run_command("python -m src.fine_tuning.test_model", "Testing fine-tuned model")
    
    if success:
        print("\n=== Pipeline completed successfully ===")
    else:
        print("\n=== Pipeline failed ===")
    
    return success

if __name__ == "__main__":
    main()
