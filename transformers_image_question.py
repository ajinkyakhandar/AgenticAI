"""Ask a vision-language model a question about an image using Transformers.

Install PyTorch if it is not already available in your environment::

    python -m pip install torch

The first run downloads the model weights from Hugging Face.
"""

from importlib.util import find_spec


# Other vision-language model IDs to try with the image-text-to-text pipeline:
# - "HuggingFaceTB/SmolVLM-500M-Instruct" (small, but larger than this example)
# - "llava-hf/llava-interleave-qwen-0.5b-hf"
# - "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
# - "google/gemma-3-4b-it" (accept Google's Hugging Face terms and log in first)
MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
IMAGE_URL = (
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"
)


def main() -> None:
    if find_spec("torch") is None:
        raise SystemExit(
            "PyTorch is required to run this vision model. Install it with: "
            "python -m pip install torch"
        )

    from transformers import pipeline

    pipe = pipeline("image-text-to-text", model=MODEL_ID)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": IMAGE_URL},
                {"type": "text", "text": "What do we see in this image?"},
            ],
        }
    ]

    result = pipe(text=messages, max_new_tokens=128, return_full_text=False)
    print(result[0]["generated_text"])


if __name__ == "__main__":
    main()
