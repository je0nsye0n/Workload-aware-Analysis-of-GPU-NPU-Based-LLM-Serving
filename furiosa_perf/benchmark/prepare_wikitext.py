"""Build fixed-length WikiText prompts using the benchmark's tokenizer."""

import argparse
import json
from pathlib import Path


def prepare(source: Path, destination: Path, tokenizer, input_tokens: int) -> int:
    """Pack source text into exact-length prompts, including tokenizer special tokens.

    Discard the final incomplete chunk and chunks that change length on decoding.
    vLLM can repeat these prompts when a scenario requests more than are available.
    """
    special_tokens = len(tokenizer.encode("", add_special_tokens=True))
    chunk_size = input_tokens - special_tokens
    if chunk_size <= 0:
        raise ValueError("input_tokens must exceed the tokenizer's special-token count")
    with source.open(encoding="utf-8") as stream:
        texts = [json.loads(line)["prompt"] for line in stream if line.strip()]
    tokens = tokenizer.encode("\n\n".join(texts), add_special_tokens=False)
    prompts = []
    for start in range(0, len(tokens) - chunk_size + 1, chunk_size):
        prompt = tokenizer.decode(tokens[start : start + chunk_size], skip_special_tokens=False)
        if len(tokenizer(prompt).input_ids) == input_tokens:
            prompts.append(prompt)
    if not prompts:
        raise ValueError(f"No exact {input_tokens}-token prompts could be built from {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for prompt in prompts:
            stream.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")
    return len(prompts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-tokens", required=True, type=int)
    args = parser.parse_args()
    from vllm.tokenizers import get_tokenizer

    tokenizer = get_tokenizer(args.model, trust_remote_code=True)
    count = prepare(args.source, args.destination, tokenizer, args.input_tokens)
    print(f"Prepared {count} prompts of {args.input_tokens} tokens: {args.destination}")


if __name__ == "__main__":
    main()
