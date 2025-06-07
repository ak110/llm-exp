#!/usr/bin/env python3
"""LiteLLMの検証コード置き場。"""

import litellm


def _main():
    # https://aws.amazon.com/bedrock/pricing/?nc1=h_ls
    # model: Price per 1,000 input tokens,
    # Amazon Nova Pro: $0.0008, $0.0032
    model = "amazon.nova-pro-v1:0"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hey, how are you?"},
        {"role": "assistant", "content": "Hey, how are you doing today?"},
        {"role": "user", "content": "Waaaah, I'm so happy to see you!"},
    ]
    output_message = "Hey, how are you doing today?"
    input_tokens = litellm.token_counter(
        model=model,
        messages=messages,
        count_response_tokens=False,  # Trueだと何かがずれる
    )
    output_tokens = litellm.token_counter(
        model=model,
        text=output_message,
        count_response_tokens=False,  # Trueだと何かがずれる
    )
    print(f"InputTokens: {input_tokens}")
    print(f"OutputTokens: {output_tokens}")

    input_cost = litellm.completion_cost(model=model, messages=messages)
    print(f"InputCost/1mTokens: {input_cost * 1e6 / input_tokens:.3f}")

    output_cost = litellm.completion_cost(model=model, completion=output_message)
    print(f"OutputCost/1mTokens: {output_cost * 1e6 / output_tokens:.3f}")

    total_cost = litellm.completion_cost(
        model=model, messages=messages, completion=output_message
    )
    print(f"InputCost+OutputCost: {input_cost + output_cost:.9f}")
    print(f"TotalCost: {total_cost:.9f}")


if __name__ == "__main__":
    _main()
