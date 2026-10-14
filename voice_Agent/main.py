import asyncio
import os
from pathlib import Path

from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer
import speech_recognition as sr
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise RuntimeError("Set OPENAI_API_KEY in the project-root .env file.")

openai_client = AsyncOpenAI(api_key=openai_api_key)


def listen_and_transcribe() -> str:
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        recognizer.pause_threshold = 2
        print("Speak something...")
        audio = recognizer.listen(source)

    print("Transcribing audio...")
    return recognizer.recognize_google(audio)


async def answer_with_openai(text: str) -> str:
    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": text}],
    )
    return response.choices[0].message.content or "I couldn't create a response."


async def speak(text: str) -> None:
    async with openai_client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        input=text,
        response_format="pcm",
    ) as response:
        await LocalAudioPlayer().play(response)


async def main() -> None:

    try:
        text = await asyncio.to_thread(listen_and_transcribe)
        print(f"You said: {text}")

        print("OpenAI is thinking...")
        reply = await answer_with_openai(text)
        print(f"OpenAI: {reply}")
        await speak(reply)

    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
    except sr.RequestError as error:
        print(f"Speech recognition service error: {error}")
    except (OSError, AttributeError) as error:
        print(f"Microphone error: {error}")
    except Exception as error:
        print(f"OpenAI voice assistant failed: {error}")
    finally:
        await openai_client.close()


if __name__ == "__main__":
    asyncio.run(main())
