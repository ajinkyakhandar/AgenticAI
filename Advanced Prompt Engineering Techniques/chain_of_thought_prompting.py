"""Chain-of-thought prompting: request a concise visible solution process."""

from common import run_prompt


def main() -> None:
    messages = [
        {
            "role": "user",
            "content": (
                "A shop has 48 notebooks at the start of the day. It sells 17 notebooks "
                "in the morning and 9 in the afternoon. Later, a delivery brings 25 more. "
                "Show the calculation in numbered steps, then state how many notebooks "
                "the shop has at the end of the day."
            ),
        }
    ]
    print(run_prompt(messages))


if __name__ == "__main__":
    main()
