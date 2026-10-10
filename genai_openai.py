import os

from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_google_ai_api_key_here":
        raise SystemExit("Set GOOGLE_API_KEY in your .env file before running this script.")

    # Use the OpenAI SDK with Google's OpenAI-compatible Gemini endpoint.
    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    response = client.chat.completions.create(
        model="gemini-3.8-flash",
        messages=[{"role": "user", "content": "Hi"}],
    )
    print(response.choices[0].message.content or "(No text response)")


if __name__ == "__main__":
    main()
