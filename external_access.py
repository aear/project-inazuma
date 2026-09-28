"""Fail-closed policy boundary for read-only external information access."""
from __future__ import annotations

from dataclasses import dataclass
import ipaddress
from typing import Any, Callable, Iterable
from urllib import error, parse, request


class ExternalAccessBlocked(RuntimeError):
    pass


@dataclass(frozen=True)
class ExternalPolicy:
    name: str
    allowed_hosts: tuple[str, ...]
    max_response_bytes: int = 512 * 1024
    timeout_seconds: float = 10.0
    max_requests: int = 1
    allowed_content_types: tuple[str, ...] = ("application/json", "text/csv", "text/plain")

    def __post_init__(self) -> None:
        if not self.name or not self.allowed_hosts:
            raise ValueError("external policy requires a name and allowed hosts")
        if not 1 <= self.max_response_bytes <= 8 * 1024 * 1024:
            raise ValueError("max_response_bytes must be 1..8388608")
        if not .1 <= self.timeout_seconds <= 30:
            raise ValueError("timeout_seconds must be 0.1..30")
        if not 1 <= self.max_requests <= 20:
            raise ValueError("max_requests must be 1..20")


def validate_external_url(url: str, policy: ExternalPolicy) -> str:
    parsed = parse.urlsplit(str(url))
    if parsed.scheme != "https":
        raise ExternalAccessBlocked("external access requires HTTPS")
    if parsed.username or parsed.password:
        raise ExternalAccessBlocked("credentials in URLs are forbidden")
    host = (parsed.hostname or "").rstrip(".").lower()
    if host not in {item.rstrip('.').lower() for item in policy.allowed_hosts}:
        raise ExternalAccessBlocked(f"host is not allowlisted by {policy.name}")
    try:
        address = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise ExternalAccessBlocked("local and non-global addresses are forbidden")
    if parsed.fragment:
        raise ExternalAccessBlocked("URL fragments are not sent externally")
    return parse.urlunsplit(("https", parsed.netloc, parsed.path or "/", parsed.query, ""))


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ExternalAccessBlocked(f"redirect blocked ({code}); target requires separate validation")


class ExternalSession:
    def __init__(self, policy: ExternalPolicy, *, opener: Callable[..., Any] | None = None) -> None:
        self.policy = policy
        self._opener = opener or request.build_opener(_NoRedirect()).open
        self.requests_used = 0

    def get(self, url: str, *, headers: dict[str, str] | None = None) -> dict[str, Any]:
        if self.requests_used >= self.policy.max_requests:
            raise ExternalAccessBlocked("external request budget exhausted")
        validated = validate_external_url(url, self.policy)
        self.requests_used += 1
        req = request.Request(validated, method="GET", headers=dict(headers or {}))
        try:
            with self._opener(req, timeout=self.policy.timeout_seconds) as response:
                status = int(getattr(response, "status", 200))
                if status in {401, 403}:
                    raise ExternalAccessBlocked(f"access denied by remote host ({status})")
                if not 200 <= status < 300:
                    raise ExternalAccessBlocked(f"unexpected HTTP status {status}")
                response_headers = getattr(response, "headers", {}) or {}
                content_type = str(response_headers.get("Content-Type", "")).split(";", 1)[0].strip().lower()
                if content_type and content_type not in self.policy.allowed_content_types:
                    raise ExternalAccessBlocked(f"content type is not allowed: {content_type}")
                try:
                    body = response.read(self.policy.max_response_bytes + 1)
                except TypeError:
                    # Compatibility for bounded in-memory test/provider adapters;
                    # the length check below remains authoritative.
                    body = response.read()
        except error.HTTPError as exc:
            if exc.code in {401, 403}:
                raise ExternalAccessBlocked(f"access denied by remote host ({exc.code})") from exc
            raise ExternalAccessBlocked(f"HTTP request failed ({exc.code})") from exc
        if len(body) > self.policy.max_response_bytes:
            raise ExternalAccessBlocked("response exceeds byte budget")
        return {
            "url": validated, "status": status, "content_type": content_type,
            "body": body, "trust": "untrusted_external_data",
            "instructions_authorized": False, "requests_used": self.requests_used,
            "request_budget": self.policy.max_requests,
        }


__all__ = ["ExternalAccessBlocked", "ExternalPolicy", "ExternalSession", "validate_external_url"]
