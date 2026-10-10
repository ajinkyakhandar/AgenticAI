import os

from dotenv import load_dotenv
from google import genai


def main() -> None:
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_google_ai_api_key_here":
        raise SystemExit("Set GOOGLE_API_KEY in your .env file before running this script.")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents="Hi",
    )
    print(response.text or "(No text response)")


if __name__ == "__main__":
    main()
