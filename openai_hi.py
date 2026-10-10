import os

from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        raise SystemExit("Set OPENAI_API_KEY in your .env file before running this script.")

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hi"}],
    )
    print(response.choices[0].message.content or "(No text response)")


if __name__ == "__main__":
    main()
