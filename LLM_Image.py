"""Generate a caption for an image using Gemini's OpenAI-compatible API."""

import os
import warnings

from dotenv import load_dotenv

# Suppress dependency/runtime warnings before importing the API client.
warnings.filterwarnings("ignore", category=Warning)

from openai import OpenAI

IMAGE_URL = "https://i.pinimg.com/736x/64/e9/7e/64e97eb0ed95476350dd7bb4332958bd.jpg"


def main() -> None:
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GOOGLE_API_KEY in your .env file before running this script.")

    # Use the OpenAI SDK with Google's OpenAI-compatible Gemini endpoint.
    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    response = client.chat.completions.create(
        model="gemini-3.8-flash",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Generate a caption for this image in about 50 words."},
                    {"type": "image_url", "image_url": {"url": IMAGE_URL}},
                ],
            }
        ],
    )
    print(response.choices[0].message.content or "(No text response)")


if __name__ == "__main__":
    main()


# Optional OpenAI version: comment out the Gemini client/request code in main()
# and uncomment this section. Set OPENAI_API_KEY in .env before using it.
#
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# response = client.chat.completions.create(
#     model="gpt-4.1-mini",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": "What's in this image?"},
#                 {"type": "image_url", "image_url": {"url": IMAGE_URL}},
#             ],
#         }
#     ],
# )
# print(response.choices[0].message.content)
