"""Shared Valkey connection and RQ queue used by the API and worker."""

from redis import Redis
from rq import Queue

# Valkey is Redis-protocol compatible; both processes use this same queue name.
connection = Redis(host="localhost", port=6379, db=0)
queue = Queue("default", connection=connection)
