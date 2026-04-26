"""Shared rate limiter instance for all routers.

Behind a load balancer / reverse proxy (Cloud Run, ALB, nginx, ...) the
direct ``request.client.host`` is always the proxy's address — every
request appears to come from the same IP and the rate limit is bypassed.

When ``TRUST_PROXY=true`` is set in the environment, the limiter reads
the first IP from the ``X-Forwarded-For`` header instead. This must NOT
be enabled in direct-connection deployments since clients can forge the
header.
"""

from __future__ import annotations

import os

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request


def _client_ip(request: Request) -> str:
    """Resolve the rate-limit key honoring proxy headers when trusted."""
    if os.getenv("TRUST_PROXY", "").lower() in ("1", "true", "yes"):
        xff = request.headers.get("X-Forwarded-For")
        if xff:
            # XFF is comma-separated: original-client, proxy1, proxy2, ...
            first = xff.split(",", 1)[0].strip()
            if first:
                return first
    return get_remote_address(request)


limiter = Limiter(key_func=_client_ip)
