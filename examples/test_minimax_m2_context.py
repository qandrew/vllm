#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Simple test/example for MinimaxM2Context usage.

This demonstrates how to use the MinimaxM2Context, MinimaxM2Parser,
and MinimaxM2Message classes for managing MiniMax M2 conversations.
"""

from vllm.entrypoints.context import (
    MinimaxM2Context,
    MinimaxM2Message,
    MinimaxM2Parser,
)


def test_minimax_m2_message():
    """Test MinimaxM2Message class."""
    print("Testing MinimaxM2Message...")

    # Create a simple message
    msg = MinimaxM2Message(
        role="assistant",
        content="Hello! How can I help you today?"
    )
    print(f"Simple message: {msg.to_dict()}")

    # Create a message with tool calls
    tool_msg = MinimaxM2Message(
        role="assistant",
        tool_calls=[{
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "San Francisco", "unit": "celsius"}'
            }
        }]
    )
    print(f"Tool call message: {tool_msg.to_dict()}")

    # Create a message with reasoning
    reasoning_msg = MinimaxM2Message(
        role="assistant",
        content="The answer is 42",
        reasoning="Let me think... the ultimate question..."
    )
    print(f"Reasoning message: {reasoning_msg.to_dict()}")

    print("✓ MinimaxM2Message tests passed\n")


def test_minimax_m2_parser():
    """Test MinimaxM2Parser class."""
    print("Testing MinimaxM2Parser...")

    parser = MinimaxM2Parser()

    # Simulate parsing text with tool call
    test_text = """<minimax:tool_call>
<invoke name="get_weather">
<parameter name="location">San Francisco</parameter>
<parameter name="unit">celsius</parameter>
</invoke>
</minimax:tool_call>"""

    parser.current_text = test_text
    parser._finalize_tool_call()

    messages = parser.finalize()
    print(f"Parsed {len(messages)} message(s)")
    for msg in messages:
        print(f"  - {msg.to_dict()}")

    print("✓ MinimaxM2Parser tests passed\n")


def test_minimax_m2_context():
    """Test MinimaxM2Context initialization."""
    print("Testing MinimaxM2Context...")

    # Create a context with initial messages
    initial_messages = [
        {"role": "user", "content": "What's the weather in SF?"}
    ]

    context = MinimaxM2Context(
        messages=initial_messages,
        available_tools=["python", "browser"],
        tokenizer=None,
    )

    print(f"Initial messages: {len(context.messages)}")
    print(f"Available tools: {context.available_tools}")
    print(f"Token usage - prompt: {context.num_prompt_tokens}, output: {context.num_output_tokens}")

    # Test need_builtin_tool_call
    assert not context.need_builtin_tool_call(), "Should not need tool call yet"

    # Add a message with tool call
    context._messages.append({
        "role": "assistant",
        "tool_calls": [{
            "function": {
                "name": "python",
                "arguments": '{"code": "print(42)"}'
            }
        }]
    })

    assert context.need_builtin_tool_call(), "Should need tool call now"

    print("✓ MinimaxM2Context tests passed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("MinimaxM2Context Test Suite")
    print("=" * 60 + "\n")

    try:
        test_minimax_m2_message()
        test_minimax_m2_parser()
        test_minimax_m2_context()

        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
