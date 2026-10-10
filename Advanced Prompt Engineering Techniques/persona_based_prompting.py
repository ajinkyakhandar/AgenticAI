"""Persona-based prompting: define expertise, audience, and tone."""

from common import run_prompt


def main() -> None:
    messages = [
        {
            "role": "system",
            "content": (
                "Act as a patient computer-science tutor helping a beginner. Use simple "
                "words, explain any technical term, and include one everyday analogy. "
                "Keep your answer under 120 words."
            ),
        },
        {
            "role": "user",
            "content": "What is a Python virtual environment? Explain why I might use one.",
        },
    ]
    print(run_prompt(messages))


if __name__ == "__main__":
    main()
