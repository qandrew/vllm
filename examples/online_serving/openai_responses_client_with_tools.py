# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Set up this example by starting a vLLM OpenAI-compatible server with tool call
options enabled.
Reasoning models can be used through the Responses API as seen here
https://platform.openai.com/docs/api-reference/responses
For example:

--tool-call-parser minimax_m2 -> this outputs <minimax:tool_call> which isn't what the model has?? <tool_calls>

vllm serve MiniMaxAI/MiniMax-M2 \
    --tensor-parallel-size 4 \
    --tool-call-parser minimax \
    --reasoning-parser minimax_m2_append_think \
    --enable-auto-tool-choice \
    --chat-template examples/tool_chat_template_minimax_m1.jinja \
    --port 8000

vllm serve Qwen/Qwen3-1.7B --reasoning-parser qwen3 \
      --structured-outputs-config.backend xgrammar \
      --enable-auto-tool-choice --tool-call-parser hermes
"""

import json

from openai import OpenAI
from utils import get_first_model


def get_horoscope(sign):
    return f"{sign}: Next Tuesday you will befriend a baby otter."


tools = [
    {
        "type": "custom",
        "name": "get_horoscope",
        "description": "Get today’s horoscope for an astrological sign.",
        "parameters": {
          "type": "object",
          "properties": {
            "sign": {
              "type": "string",
              "description": "Astrological sign, e.g. Aries, Taurus, Gemini, etc."
            }
          },
          "required": ["sign"]
        }
      }
]

input_messages = [
    {"role": "user", "content": "What is the horoscope for Leo today?"}
]


def main():
    base_url = "http://localhost:8000/v1"
    client = OpenAI(base_url=base_url, api_key="empty")
    model = get_first_model(client)
    response = client.responses.create(
        model=model, input=input_messages, tools=tools,
        # tool_choice="required" #this breaks it for custom tools
    )

    print(response)
    # import fbvscode; fbvscode.set_trace()

    for out in response.output:
        if out.type == "function_call":
            print("Function call:", out.name, out.arguments)
            tool_call = out
    args = json.loads(tool_call.arguments)
    result = get_horoscope(args["sign"])

    input_messages.append(tool_call)  # append model's function call message
    input_messages.append(
        {  # append result message
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": str(result),
        }
    )
    response_2 = client.responses.create(
        model=model,
        input=input_messages,
        tools=tools,
    )
    print(response_2.output_text)


if __name__ == "__main__":
    main()
