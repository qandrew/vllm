# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

from openai import OpenAI

"""
https://huggingface.co/MiniMaxAI/MiniMax-M2/blob/main/docs/tool_calling_guide.md

vllm serve MiniMaxAI/MiniMax-M2 \
    --tensor-parallel-size 4 \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think \
    --enable-auto-tool-choice \
    --port 8000
"""

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")


def get_weather(location: str, unit: str):
    return f"The weather for {location} in {unit} is 20"


tool_functions = {"get_weather": get_weather}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and state, e.g., 'San Francisco, CA'",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location", "unit"],
            },
        },
    }
]

messages = [
    {
        "role": "user",
        "content": "What's the weather like in San Francisco? use celsius.",
    }
]

response = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

# print(response)

# tool_call = response.choices[0].message.tool_calls[0].function
# print(f"Function called: {tool_call.name}")
# print(f"Arguments: {tool_call.arguments}")
# print(f"Result: {get_weather(**json.loads(tool_call.arguments))}")

# feed back into
# import fbvscode; fbvscode.set_trace()

print("=== First response ===")
print(response)

# Step 3: Extract and call the function
tool_call = response.choices[0].message.tool_calls[0].function
name = tool_call.name
args = json.loads(tool_call.arguments)
result = tool_functions[name](**args)

print(f"\nFunction called: {name}")
print(f"Arguments: {args}")
print(f"Result: {result}")

# Step 4: Send the result back to the model
messages.append(
    {"role": "assistant", "tool_calls": response.choices[0].message.tool_calls}
)
messages.append(
    {
        "role": "tool",
        "tool_call_id": response.choices[0].message.tool_calls[0].id,
        "content": result,
    }
)

# Step 5: Second call — model sees tool output
second_response = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=messages,
)

print("\n=== Second response ===")
print(second_response.choices[0].message)
