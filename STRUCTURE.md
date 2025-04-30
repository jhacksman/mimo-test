# Project Structure and Implementation Details

This document provides detailed information about the implementation of the MiMo-7B tool-calling fine-tuning project.

## Overview

The project fine-tunes the Xiaomi MiMo-7B-RL model to add tool-calling capabilities. The process involves:

1. **Data Generation**: Using Gemini 2.5 Pro via OpenRouter to create a dataset of examples suitable for teaching tool use.
2. **Data Validation**: Using Pydantic (a Python implementation similar to Zod) to validate the structure and data types of the generated examples.
3. **Fine-tuning**: Using the validated dataset to fine-tune the MiMo-7B-RL model via Supervised Fine-Tuning (SFT).

## Data Generation

The data generation process uses Gemini 2.5 Pro via OpenRouter to create examples of tool-calling conversations. The process:

1. Defines a set of tools that the model should learn to call
2. Prompts Gemini 2.5 Pro to generate realistic user queries and corresponding assistant responses that call the appropriate tools
3. Parses the generated content into a structured format suitable for training

### Tool Definitions

The tools defined for this project include:

- `get_weather`: Get weather information for a location
- `search_database`: Search a database for information
- `calculate`: Perform a calculation

Each tool has a name, description, and parameters with their types and descriptions.

### Generated Data Format

The generated data follows this format:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What's the weather like in New York?"
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_1",
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": "{\"location\":\"New York\",\"unit\":\"celsius\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_1",
      "name": "get_weather",
      "content": "The current weather in New York is 22°C and partly cloudy."
    },
    {
      "role": "assistant",
      "content": "The current weather in New York is 22°C (celsius) and partly cloudy."
    }
  ],
  "tools": [
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
}
```

## Data Validation

The data validation process uses Pydantic to ensure that the generated examples have the correct structure and data types. The process:

1. Defines schemas for different message types (user, assistant, tool)
2. Validates each message in the generated examples
3. Ensures that each example has at least one valid tool call
4. Outputs a cleaned dataset suitable for fine-tuning

## Fine-tuning

The fine-tuning process uses the Hugging Face Transformers library to fine-tune the MiMo-7B-RL model. The process:

1. Loads the pre-trained MiMo-7B-RL model
2. Prepares the validated dataset for training
3. Uses 4-bit quantization and LoRA for efficient fine-tuning
4. Trains the model to recognize when to call tools and how to format the tool calls

### Training Format

The training examples are formatted as follows:

```
<s>[INST] You have access to the following tools:
{tool_definition_json}

{user_message} [/INST] I'll help you with that. I need to use the {tool_name} tool.
<tool_call>
{"name": "{tool_name}", "arguments": {tool_arguments}}
</tool_call>
<tool_result>
{"tool_call_id": "{tool_call_id}", "name": "{tool_name}", "content": "{tool_result}"}
</tool_result>
[INST] {follow_up_user_message} [/INST] {assistant_response}</s>
```

## Testing

The testing process verifies that the fine-tuned model can correctly generate tool calls. The process:

1. Loads the fine-tuned model
2. Creates a test prompt with a tool definition and a user query
3. Generates a response from the model
4. Checks if the response contains a properly formatted tool call

## Hardware Considerations

The fine-tuning process is optimized to work within a 64GB VRAM constraint using:

1. **4-bit Quantization**: Reduces the memory footprint of the model
2. **LoRA**: Efficient fine-tuning that updates only a small number of parameters
3. **Gradient Accumulation**: Simulates larger batch sizes by accumulating gradients over multiple forward passes

These optimizations allow the fine-tuning process to run on consumer-grade hardware while still producing good results.
