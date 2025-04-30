"""Test the fine-tuned model's tool-calling capabilities."""

import json
import os
from pathlib import Path
from typing import Dict, List, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import MODEL_DIR, OUTPUT_MODEL_ID

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g., San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

def test_model():
    """Test the fine-tuned model with a sample prompt."""
    model_path = MODEL_DIR / OUTPUT_MODEL_ID
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return
    
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    prompt = "<s>[INST] "
    prompt += "You have access to the following tools:\n"
    prompt += json.dumps(TOOLS[0], indent=2) + "\n\n"
    prompt += "What's the weather like in New York today? [/INST]"
    
    print("Testing model with prompt:")
    print(prompt)
    print("\nGenerating response...")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    
    print("\nModel response:")
    print(response)
    
    if "<tool_call>" in response and "</tool_call>" in response:
        print("\nSuccess! The model generated a tool call.")
        
        tool_call_start = response.find("<tool_call>") + len("<tool_call>")
        tool_call_end = response.find("</tool_call>")
        tool_call = response[tool_call_start:tool_call_end].strip()
        
        print("\nTool call:")
        print(tool_call)
    else:
        print("\nThe model did not generate a tool call.")

if __name__ == "__main__":
    test_model()
