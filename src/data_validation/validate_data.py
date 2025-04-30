"""Validate and clean the generated training data using Zod-inspired validation."""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

from pydantic import BaseModel, Field, validator

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import DATA_DIR

class FunctionCall(BaseModel):
    name: str
    arguments: str  # JSON string

class ToolCall(BaseModel):
    id: str
    type: str
    function: FunctionCall

class BaseMessage(BaseModel):
    role: str

class UserMessage(BaseMessage):
    content: str
    
    @validator('role')
    def validate_role(cls, v):
        if v != 'user':
            raise ValueError('Role must be "user"')
        return v

class AssistantMessageWithContent(BaseMessage):
    content: str
    
    @validator('role')
    def validate_role(cls, v):
        if v != 'assistant':
            raise ValueError('Role must be "assistant"')
        return v

class AssistantMessageWithToolCalls(BaseMessage):
    content: Optional[str] = None
    tool_calls: List[ToolCall]
    
    @validator('role')
    def validate_role(cls, v):
        if v != 'assistant':
            raise ValueError('Role must be "assistant"')
        return v

class ToolMessage(BaseMessage):
    content: str
    tool_call_id: str
    name: str
    
    @validator('role')
    def validate_role(cls, v):
        if v != 'tool':
            raise ValueError('Role must be "tool"')
        return v

class TrainingExample(BaseModel):
    messages: List[Any]  # We'll validate this separately
    tools: List[Dict[str, Any]]
    functions: List[Dict[str, Any]] = []

def validate_and_clean_examples(examples_file: Path) -> List[Dict[str, Any]]:
    """Validate and clean the generated examples."""
    with open(examples_file, 'r') as f:
        raw_examples = json.load(f)
    
    validated_examples = []
    
    for i, example in enumerate(raw_examples):
        try:
            if "parsed" in example and not example["parsed"]:
                from src.data_generation.parse_content import parse_generated_content
                example = parse_generated_content(example["content"])
            
            validated_messages = []
            for msg in example["messages"]:
                try:
                    if msg["role"] == "user":
                        validated_messages.append(UserMessage(**msg).dict())
                    elif msg["role"] == "assistant":
                        if "tool_calls" in msg and msg["tool_calls"]:
                            validated_messages.append(AssistantMessageWithToolCalls(**msg).dict())
                        else:
                            validated_messages.append(AssistantMessageWithContent(**msg).dict())
                    elif msg["role"] == "tool":
                        validated_messages.append(ToolMessage(**msg).dict())
                except Exception as e:
                    print(f"Invalid message in example {i+1}: {str(e)}")
                    continue
            
            has_tool_call = any("tool_calls" in msg for msg in validated_messages)
            if has_tool_call:
                from src.data_generation.generate_data import TOOLS
                
                validated_example = TrainingExample(
                    messages=validated_messages,
                    tools=TOOLS,
                    functions=[]  # Legacy format
                ).dict()
                
                validated_examples.append(validated_example)
                print(f"Validated example {i+1}")
            else:
                print(f"Skipping example {i+1}: No tool calls found")
                
        except Exception as e:
            print(f"Error validating example {i+1}: {str(e)}")
            continue
    
    return validated_examples

if __name__ == "__main__":
    raw_data_file = DATA_DIR / "raw" / "tool_calling_examples_raw.json"
    
    if not raw_data_file.exists():
        print(f"Raw data file not found: {raw_data_file}")
        exit(1)
    
    print(f"Validating examples from {raw_data_file}...")
    validated_examples = validate_and_clean_examples(raw_data_file)
    
    validated_dir = DATA_DIR / "validated"
    validated_dir.mkdir(exist_ok=True, parents=True)
    
    validated_file = validated_dir / "tool_calling_examples_validated.json"
    with open(validated_file, "w") as f:
        json.dump(validated_examples, f, indent=2)
    
    print(f"Validation complete. Saved {len(validated_examples)} valid examples to {validated_file}")
