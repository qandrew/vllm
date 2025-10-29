from openai import OpenAI
import json

"""
https://huggingface.co/MiniMaxAI/MiniMax-M2/blob/main/docs/tool_calling_guide.md

vllm serve MiniMaxAI/MiniMax-M2 \
    --tensor-parallel-size 4 \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think
    --structured-outputs-config.backend xgrammar \
    --enable-auto-tool-choice \
    --port 8000
"""

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")


def get_weather(location: str, unit: str):
    return f"Getting the weather for {location} in {unit}..."


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

response = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[
        {
            "role": "user",
            "content": "What's the weather like in San Francisco? use celsius.",
        }
    ],
    tools=tools,
    tool_choice="auto",
)

print(response)

tool_call = response.choices[0].message.tool_calls[0].function
print(f"Function called: {tool_call.name}")
print(f"Arguments: {tool_call.arguments}")
print(f"Result: {get_weather(**json.loads(tool_call.arguments))}")
