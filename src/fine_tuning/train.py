"""Fine-tune the MiMo-7B-RL model for tool-calling capabilities."""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Any

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import DATA_DIR, MODEL_DIR, OUTPUT_DIR, BASE_MODEL_ID, OUTPUT_MODEL_ID

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(f"{OUTPUT_DIR}/training.log")],
)
logger = logging.getLogger(__name__)

TRAINING_BATCH_SIZE = int(os.environ.get("TRAINING_BATCH_SIZE", "4"))

def prepare_dataset(data_file: Path) -> Dataset:
    """Prepare the dataset for fine-tuning."""
    with open(data_file, "r") as f:
        data = json.load(f)
    
    training_samples = []
    
    for example in data:
        formatted_prompt = format_conversation_for_training(example["messages"], example["tools"])
        training_samples.append({"text": formatted_prompt})
    
    return Dataset.from_list(training_samples)

def format_conversation_for_training(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> str:
    """Format a conversation for fine-tuning."""
    
    formatted = "<s>[INST] "
    
    if tools:
        formatted += "You have access to the following tools:\n"
        for tool in tools:
            formatted += json.dumps(tool, indent=2) + "\n"
        formatted += "\n"
    
    for i, message in enumerate(messages):
        if message["role"] == "user":
            if i > 0:  # Not the first message
                formatted += "[/INST] [INST] "
            formatted += message["content"] + " "
        elif message["role"] == "assistant":
            if "tool_calls" in message and message["tool_calls"]:
                formatted += "[/INST] "
                for tool_call in message["tool_calls"]:
                    formatted += f"I'll help you with that. I need to use the {tool_call['function']['name']} tool.\n"
                    formatted += f"<tool_call>\n"
                    formatted += f"{{\"name\": \"{tool_call['function']['name']}\", \"arguments\": {tool_call['function']['arguments']}}}\n"
                    formatted += f"</tool_call>\n"
            elif message["content"]:
                formatted += "[/INST] " + message["content"] + " "
        elif message["role"] == "tool":
            formatted += f"<tool_result>\n"
            formatted += f"{{\"tool_call_id\": \"{message['tool_call_id']}\", \"name\": \"{message['name']}\", \"content\": \"{message['content']}\"}}\n"
            formatted += f"</tool_result>\n"
            formatted += "[INST] "
    
    if not formatted.endswith("[INST] "):
        formatted += "</s>"
    else:
        formatted = formatted[:-7] + "</s>"
    
    return formatted

def train_model():
    """Fine-tune the MiMo-7B-RL model for tool-calling capabilities."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if device.type == "cuda":
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        logger.info(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    data_file = DATA_DIR / "validated" / "tool_calling_examples_validated.json"
    logger.info(f"Loading dataset from {data_file}")
    dataset = prepare_dataset(data_file)
    logger.info(f"Dataset size: {len(dataset)}")
    
    logger.info(f"Loading tokenizer from {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loading model from {BASE_MODEL_ID} with 4-bit quantization")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )
    
    logger.info("Preparing model for LoRA fine-tuning")
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=16,                    # Rank of the update matrices
        lora_alpha=32,           # Parameter scaling factor
        lora_dropout=0.05,       # Dropout probability
        bias="none",             # Don't train bias parameters
        task_type="CAUSAL_LM",   # Task type for causal language modeling
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Target attention modules
    )
    
    model = get_peft_model(model, lora_config)
    logger.info("Applied LoRA to the model")
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=TRAINING_BATCH_SIZE,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        weight_decay=0.01,
        fp16=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        report_to=None,
        remove_unused_columns=False,
    )
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    logger.info("Starting training")
    trainer.train()
    
    logger.info("Saving fine-tuned model")
    trainer.save_model(MODEL_DIR / OUTPUT_MODEL_ID)
    tokenizer.save_pretrained(MODEL_DIR / OUTPUT_MODEL_ID)
    logger.info(f"Model saved to {MODEL_DIR / OUTPUT_MODEL_ID}")

if __name__ == "__main__":
    train_model()
