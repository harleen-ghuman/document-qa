from contextlib import contextmanager
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar(
    "request_id",
    default="-",
)


@contextmanager
def bind_request_id(request_id: str):
    token = request_id_var.set(request_id)
    try:
        yield
    finally:
        request_id_var.reset(token)