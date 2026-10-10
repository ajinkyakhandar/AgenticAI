"""Zero-shot prompting: ask for a task without providing examples."""

from common import run_prompt


def main() -> None:
    messages = [
        {
            "role": "user",
            "content": (
                "Read the customer feedback below. First, write a one-sentence summary. "
                "Then choose one overall sentiment: positive, neutral, or negative. "
                "Include both the summary and the sentiment in your answer.\n\n"
                'Customer feedback: "The headphones sound excellent, but the ear cups '
                'became uncomfortable after an hour."'
            ),
        }
    ]
    print(run_prompt(messages))


if __name__ == "__main__":
    main()
