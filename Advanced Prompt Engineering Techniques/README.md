# Advanced Prompt Engineering Techniques

Each Python script demonstrates one prompting technique with a live OpenAI chat completion. The examples share `common.py` for API-key loading and response handling.

## Setup

Install the workspace dependencies from `requirement.txt` and set `OPENAI_API_KEY` in a `.env` file in the workspace root. Optionally set `OPENAI_MODEL`; it defaults to `gpt-4o-mini`.

Run a script from the workspace root, for example:

```sh
python "Advanced Prompt Engineering Techniques/zero_shot_prompting.py"
```

## Examples

- `zero_shot_prompting.py` — gives the model a task without demonstrations.
- `one_shot_prompting.py` — includes one input/output example to establish the pattern.
- `few_shot_prompting.py` — includes several examples to clarify a classification task.
- `structured_output_few_shot.py` — combines examples with a strict JSON Schema response format.
- `persona_based_prompting.py` — assigns a role, audience, and communication style.
- `chain_of_thought_prompting.py` — asks for a concise, visible step-by-step solution.
