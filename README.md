# MiMo-7B Tool-Calling Fine-Tuning

This project fine-tunes the Xiaomi MiMo-7B-RL model to add tool-calling capabilities. The process involves:

1. Generating training data using Gemini 2.5 Pro via OpenRouter
2. Validating the data using Zod-inspired schema validation
3. Fine-tuning the MiMo-7B-RL model using the validated data

## Requirements

- Python 3.8+
- CUDA-compatible GPU with at least 24GB VRAM (recommended: 64GB)
- OpenRouter API key for accessing Gemini 2.5 Pro

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jhacksman/mimo-test.git
cd mimo-test
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.template .env
# Edit .env to add your OpenRouter API key
```

## Project Structure

```
mimo-test/
├── config.py                  # Project configuration
├── run_pipeline.py            # Main script to run the pipeline
├── requirements.txt           # Python dependencies
├── .env.template              # Template for environment variables
├── data/                      # Data directory (created during execution)
│   ├── raw/                   # Raw generated data
│   └── validated/             # Validated data for training
├── models/                    # Model directory (created during execution)
├── output/                    # Training output directory
└── src/                       # Source code
    ├── data_generation/       # Data generation scripts
    │   ├── generate_data.py   # Generate data using Gemini 2.5 Pro
    │   └── parse_content.py   # Parse generated content
    ├── data_validation/       # Data validation scripts
    │   └── validate_data.py   # Validate data using Pydantic
    └── fine_tuning/           # Fine-tuning scripts
        ├── train.py           # Fine-tune the model
        └── test_model.py      # Test the fine-tuned model
```

## Usage

### Running the Complete Pipeline

To run the complete pipeline (data generation, validation, fine-tuning, and testing):

```bash
python run_pipeline.py
```

### Running Individual Steps

To run individual steps of the pipeline:

```bash
# Generate training data
python -m src.data_generation.generate_data

# Validate training data
python -m src.data_validation.validate_data

# Fine-tune the model
python -m src.fine_tuning.train

# Test the fine-tuned model
python -m src.fine_tuning.test_model
```

### Pipeline Options

The `run_pipeline.py` script accepts several options:

```bash
# Skip data generation
python run_pipeline.py --skip-data-gen

# Skip data validation
python run_pipeline.py --skip-validation

# Skip model training
python run_pipeline.py --skip-training

# Only run model testing
python run_pipeline.py --test-only
```

## Hardware Considerations

Fine-tuning the MiMo-7B model requires significant GPU resources. The code is optimized to work within a 64GB VRAM constraint using:

- 4-bit quantization
- LoRA (Low-Rank Adaptation) for efficient fine-tuning
- Gradient accumulation

Adjust the `TRAINING_BATCH_SIZE` in the `.env` file based on your available VRAM.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
