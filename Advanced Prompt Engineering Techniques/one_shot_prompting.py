"""One-shot prompting: provide one example before the new task."""

from common import run_prompt


def main() -> None:
    messages = [
        {
            "role": "system",
            "content": (
                "Rewrite each status update as a calm, clear message for a customer. "
                "Keep the important facts, avoid blame, and do not add promises that "
                "are not in the original update. The first exchange is an example; "
                "rewrite the update in the final message."
            ),
        },
        {
            "role": "user",
            "content": 'Example status update: "The server crashed again. We are looking into it."',
        },
        {
            "role": "assistant",
            "content": "We are investigating a server issue. We will share another update when we have more information.",
        },
        {
            "role": "user",
            "content": 'Please rewrite this status update: "We found the bug. The patch is being tested and should go out this afternoon."',
        },
    ]
    print(run_prompt(messages))


if __name__ == "__main__":
    main()
