"""Ask the local Ollama llama3.1:latest model through a FastAPI endpoint.

Setup (run these commands in separate terminals where noted)::

    ollama pull llama3.1:latest
    ollama serve  # Start Ollama if it is not already running.
    python -m uvicorn integrating_ollama_with_fastapi:app --reload

Then send a question to http://127.0.0.1:8000/ask, for example::

    curl -X POST http://127.0.0.1:8000/ask \\
      -H "Content-Type: application/json" \\
      -d '{"question":"Why is the sky blue?"}'
"""

import ollama
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn


MODEL_NAME = "llama3.1:latest"

app = FastAPI(title="Ollama Q&A API")


class AskRequest(BaseModel):
    """Question sent to the local language model."""

    question: str = Field(min_length=1, examples=["Why is the sky blue?"])


class AskResponse(BaseModel):
    """Answer returned by the language model."""

    question: str
    answer: str
    model: str


@app.post("/ask", response_model=AskResponse)
def ask_ollama(request: AskRequest) -> AskResponse:
    """Send a question to Ollama and return the model's answer."""
    try:
        result = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": request.question}],
        )
    except ollama.RequestError as error:
        raise HTTPException(
            status_code=503,
            detail=f"Could not connect to Ollama. Make sure it is running: {error}",
        ) from error
    except ollama.ResponseError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama could not generate a response: {error.error}",
        ) from error

    return AskResponse(
        question=request.question,
        answer=result.message.content,
        model=MODEL_NAME,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)