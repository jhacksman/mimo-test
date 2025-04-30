"""Generate tool-calling training data using Gemini 2.5 Pro via OpenRouter."""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import openai
from openai import OpenAI
import requests

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import DATA_DIR, NUM_EXAMPLES

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set")

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
    },
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Search a database for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform a calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

def generate_tool_calling_examples(num_examples: int) -> List[Dict[str, Any]]:
    """Generate examples of tool calling conversations using Gemini 2.5 Pro."""
    examples = []
    
    for i in range(num_examples):
        try:
            response = client.chat.completions.create(
                model="google/gemini-2.5-pro",  # Using Gemini 2.5 Pro via OpenRouter
                messages=[
                    {
                        "role": "system",
                        "content": "Generate a realistic user query that would benefit from using one of the provided tools. Then generate the corresponding assistant response that properly calls the relevant tool in the exact JSON format required. Include the complete conversation from user input to final assistant response after receiving tool output."
                    },
                    {
                        "role": "user",
                        "content": f"Tools available: {json.dumps(TOOLS, indent=2)}"
                    }
                ],
                temperature=0.7,  # Add some variety to the generated examples
            )
            
            generated_content = response.choices[0].message.content
            
            example = parse_generated_content(generated_content)
            examples.append(example)
            
            print(f"Generated example {i+1}/{num_examples}")
            
            time.sleep(1)
            
        except Exception as e:
            print(f"Error generating example {i+1}: {str(e)}")
            continue
    
    return examples

def parse_generated_content(content: str) -> Dict[str, Any]:
    """Parse the generated content into a structured example.
    This function will need to be developed based on the actual output format.
    """
    return {
        "content": content,
        "parsed": False  # Flag to indicate this needs further processing
    }

if __name__ == "__main__":
    output_dir = DATA_DIR / "raw"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Generating {NUM_EXAMPLES} tool-calling examples...")
    examples = generate_tool_calling_examples(NUM_EXAMPLES)
    
    output_path = output_dir / "tool_calling_examples_raw.json"
    with open(output_path, "w") as f:
        json.dump(examples, f, indent=2)
    
    print(f"Generated {len(examples)} examples. Saved to {output_path}")
