#!/usr/bin/env python3
"""LiteLLMの検証コード置き場。"""

import litellm


def _main():
    # https://aws.amazon.com/bedrock/pricing/?nc1=h_ls
    # model: Price per 1,000 input tokens,
    # Amazon Nova Pro: $0.0008, $0.0032
    model = "amazon.nova-pro-v1:0"

    messages = [{"user": "role", "content": "Hey, how"}]
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
    print(f"OutputTokens: {input_tokens}")

    input_cost = litellm.completion_cost(model=model, messages=messages)
    print(f"InputCost/1mTokens: {input_cost * 1e6 / input_tokens}")

    output_cost = litellm.completion_cost(model=model, completion=output_message)
    print(f"OutputCost/1mTokens: {output_cost * 1e6 / output_tokens}")

    total_cost = litellm.completion_cost(
        model=model, messages=messages, completion=output_message
    )
    print(f"InputCost+OutputCost: {input_cost + output_cost}")
    print(f"TotalCost: {total_cost}")


if __name__ == "__main__":
    _main()
