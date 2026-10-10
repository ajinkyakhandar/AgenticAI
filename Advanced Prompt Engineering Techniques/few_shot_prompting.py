"""Few-shot prompting: show several examples of a classification pattern."""

from common import run_prompt


def main() -> None:
    messages = [
        {
            "role": "system",
            "content": (
                "Classify each product review as positive, neutral, or negative. "
                "Positive means the customer is satisfied. Negative means the customer "
                "is dissatisfied. Neutral means the review gives information without "
                "clear praise or complaint. Study the three example reviews and labels, "
                "then classify the final review. Reply with only one label."
            ),
        },
        {"role": "user", "content": "Example review 1: The delivery was quick, and the headphones work perfectly."},
        {"role": "assistant", "content": "positive"},
        {"role": "user", "content": "Example review 2: The package arrived today. It contains a blue mug."},
        {"role": "assistant", "content": "neutral"},
        {"role": "user", "content": "Example review 3: The mug arrived broken, and customer support has not replied."},
        {"role": "assistant", "content": "negative"},
        {"role": "user", "content": "Now classify this review: Setup was easy, and the app has worked reliably all week."},
    ]
    print(run_prompt(messages))


if __name__ == "__main__":
    main()
