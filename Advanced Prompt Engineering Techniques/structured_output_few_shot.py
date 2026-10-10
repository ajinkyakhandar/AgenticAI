"""Combine few-shot examples with a strict JSON Schema response format."""

from common import run_prompt


TASK_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "task_summary",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                "action_items": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["summary", "priority", "action_items"],
            "additionalProperties": False,
        },
    },
}


def main() -> None:
    messages = [
        {
            "role": "system",
            "content": (
                "Read each work note and return three things: a short summary, a priority, "
                "and a list of action items. Use high priority for urgent issues affecting "
                "customers, medium for normal deadlines, and low for non-urgent work. "
                "Copy the style of the two examples. The final answer must follow the "
                "provided JSON schema exactly."
            ),
        },
        {"role": "user", "content": "Example note 1: Sam will draft the launch email tomorrow. The draft is due Friday; this is routine work."},
        {"role": "assistant", "content": '{"summary":"The launch email draft is due Friday.","priority":"medium","action_items":["Sam drafts the email tomorrow"]}'},
        {"role": "user", "content": "Example note 2: The checkout service is failing for many customers. Maya must investigate now and report back in one hour."},
        {"role": "assistant", "content": '{"summary":"Many customers are affected by checkout failures.","priority":"high","action_items":["Maya investigates now","Maya reports back in one hour"]}'},
        {"role": "user", "content": "Now process this note: Customers cannot log in after today's update. Lee should reproduce the issue today and send a proposed fix before tomorrow's review."},
    ]
    print(run_prompt(messages, response_format=TASK_SCHEMA))


if __name__ == "__main__":
    main()
