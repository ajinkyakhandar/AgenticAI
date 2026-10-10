# /// script
# requires-python = ">=3.9"
# dependencies = ["tiktoken>=0.14.0"]
# # Add dependency requirement strings to the array above as needed.
# ///

import tiktoken


def decode_tokens(token_ids: list[int]) -> str:
    """Decode token IDs back into text using the same encoding."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return encoding.decode(token_ids)


def main() -> None:
    text = "Hello, tiktoken!"
    encoding = tiktoken.get_encoding("cl100k_base")
    # Convert the input text into token IDs, then decode them back to text.
    tokens = encoding.encode(text)
    decoded_text = decode_tokens(tokens)

    print(f"Input text (original string): {text}")
    print(f"Token IDs (integer representation): {tokens}")
    print(f"Token count (number of tokens): {len(tokens)}")
    print(f"Decoded text (reconstructed string): {decoded_text}")


# Expected output:
# Input text (original string): Hello, tiktoken!
# Token IDs (integer representation): [9906, 11, 87272, 5963, 0]
# Token count (number of tokens): 5
# Decoded text (reconstructed string): Hello, tiktoken!
if __name__ == "__main__":
    main()
