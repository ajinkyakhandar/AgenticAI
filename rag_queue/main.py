"""FastAPI service for submitting RAG queries to the RQ worker."""

from fastapi import FastAPI, HTTPException, Query
from redis.exceptions import RedisError
from rq.exceptions import NoSuchJobError
from rq.job import Job
from client.rq_client import connection, queue
import uvicorn

app = FastAPI(title="RAG Queue API")


@app.get("/chat", status_code=202)
def submit_query(query: str = Query(..., min_length=1, description="Question to ask about the indexed PDF")) -> dict[str, str]:
    """Enqueue a RAG query and return its RQ job ID."""
    try:
        job = queue.enqueue("queues.worker.process_queue", query)
    except RedisError as exc:
        raise HTTPException(status_code=503, detail="Could not enqueue query; Valkey is unavailable") from exc
    return {"job_id": job.id, "status": "queued"}


@app.get("/result/{job_id}")
def get_result(job_id: str) -> dict[str, object]:
    """Return the status and, when ready, result for an enqueued query."""
    try:
        job = Job.fetch(job_id, connection=connection)
    except NoSuchJobError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    except RedisError as exc:
        raise HTTPException(status_code=503, detail="Valkey is unavailable") from exc

    status = job.get_status(refresh=True)
    response: dict[str, object] = {"job_id": job.id, "status": status}
    if status == "finished":
        response["result"] = job.result
    elif status == "failed":
        response["error"] = job.exc_info
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
