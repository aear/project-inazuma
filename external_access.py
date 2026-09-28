"""Fail-closed policy boundary for read-only external information access."""
from __future__ import annotations

from dataclasses import dataclass
import http.client
import ipaddress
import socket
import ssl
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
    allowed_methods: tuple[str, ...] = ("GET",)

    def __post_init__(self) -> None:
        if not self.name or not self.allowed_hosts:
            raise ValueError("external policy requires a name and allowed hosts")
        if not 1 <= self.max_response_bytes <= 8 * 1024 * 1024:
            raise ValueError("max_response_bytes must be 1..8388608")
        if not .1 <= self.timeout_seconds <= 30:
            raise ValueError("timeout_seconds must be 0.1..30")
        if not 1 <= self.max_requests <= 20:
            raise ValueError("max_requests must be 1..20")
        if not self.allowed_methods or any(method not in {"GET", "POST"} for method in self.allowed_methods):
            raise ValueError("allowed_methods must contain only GET or POST")


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


def resolve_public_addresses(host: str, *, resolver: Callable[..., Any] = socket.getaddrinfo) -> tuple[str, ...]:
    try:
        answers = resolver(host, 443, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ExternalAccessBlocked("DNS resolution failed") from exc
    addresses = []
    for answer in answers:
        try:
            address = str(answer[4][0])
            parsed = ipaddress.ip_address(address)
        except (IndexError, ValueError, TypeError):
            raise ExternalAccessBlocked("DNS returned an invalid address")
        if not parsed.is_global:
            raise ExternalAccessBlocked(f"DNS returned non-public address: {address}")
        canonical = parsed.compressed
        if canonical not in addresses:
            addresses.append(canonical)
    if not addresses:
        raise ExternalAccessBlocked("DNS returned no usable addresses")
    return tuple(addresses)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, hostname: str, address: str, *, port: int, timeout: float) -> None:
        super().__init__(hostname, port=port, timeout=timeout, context=ssl.create_default_context())
        self._pinned_address = address

    def connect(self) -> None:
        raw = socket.create_connection((self._pinned_address, self.port), self.timeout)
        peer = ipaddress.ip_address(raw.getpeername()[0]).compressed
        if peer != ipaddress.ip_address(self._pinned_address).compressed:
            raw.close()
            raise ExternalAccessBlocked("connected peer differs from pinned DNS address")
        self.sock = self._context.wrap_socket(raw, server_hostname=self.host)


class ExternalSession:
    def __init__(
        self, policy: ExternalPolicy, *, opener: Callable[..., Any] | None = None,
        resolver: Callable[..., Any] = socket.getaddrinfo,
    ) -> None:
        self.policy = policy
        self._opener = opener
        self._resolver = resolver
        self.requests_used = 0

    def get(self, url: str, *, headers: dict[str, str] | None = None) -> dict[str, Any]:
        return self.request("GET", url, headers=headers)

    def request(
        self, method: str, url: str, *, headers: dict[str, str] | None = None,
        body: bytes | None = None,
    ) -> dict[str, Any]:
        selected_method = str(method).upper()
        if selected_method not in self.policy.allowed_methods:
            raise ExternalAccessBlocked(f"HTTP method is not allowed: {selected_method}")
        if self.requests_used >= self.policy.max_requests:
            raise ExternalAccessBlocked("external request budget exhausted")
        validated = validate_external_url(url, self.policy)
        self.requests_used += 1
        parsed = parse.urlsplit(validated)
        addresses = (
            resolve_public_addresses(parsed.hostname or "", resolver=self._resolver)
            if self._opener is None else ()
        )
        destination_verified = False
        connected_address = None
        if self._opener is not None:
            req = request.Request(validated, data=body, method=selected_method, headers=dict(headers or {}))
            try:
                with self._opener(req, timeout=self.policy.timeout_seconds) as response:
                    status = int(getattr(response, "status", 200))
                    response_headers = getattr(response, "headers", {}) or {}
                    try:
                        response_body = response.read(self.policy.max_response_bytes + 1)
                    except TypeError:
                        response_body = response.read()
                    destination_verified = bool(getattr(response, "destination_verified", False))
                    connected_address = getattr(response, "connected_address", None)
                    injected_addresses = getattr(response, "resolved_addresses", ())
                    if injected_addresses:
                        addresses = tuple(str(item) for item in injected_addresses)
            except error.HTTPError as exc:
                if exc.code in {401, 403}:
                    raise ExternalAccessBlocked(f"access denied by remote host ({exc.code})") from exc
                raise ExternalAccessBlocked(f"HTTP request failed ({exc.code})") from exc
        else:
            connected_address = addresses[0]
            connection = _PinnedHTTPSConnection(
                parsed.hostname or "", connected_address, port=parsed.port or 443,
                timeout=self.policy.timeout_seconds,
            )
            target = parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
            outbound_headers = {"Host": parsed.hostname or "", "Connection": "close", **dict(headers or {})}
            try:
                connection.request(selected_method, target, body=body, headers=outbound_headers)
                response = connection.getresponse()
                status = int(response.status)
                response_headers = response.headers
                response_body = response.read(self.policy.max_response_bytes + 1)
                destination_verified = True
            except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
                raise ExternalAccessBlocked("pinned HTTPS request failed") from exc
            finally:
                connection.close()
        if 300 <= status < 400:
            raise ExternalAccessBlocked("redirect blocked; destination was not followed")
        if status in {401, 403}:
            raise ExternalAccessBlocked(f"access denied by remote host ({status})")
        if not 200 <= status < 300:
            raise ExternalAccessBlocked(f"unexpected HTTP status {status}")
        content_type = str(response_headers.get("Content-Type", "")).split(";", 1)[0].strip().lower()
        if content_type and content_type not in self.policy.allowed_content_types:
            raise ExternalAccessBlocked(f"content type is not allowed: {content_type}")
        if len(response_body) > self.policy.max_response_bytes:
            raise ExternalAccessBlocked("response exceeds byte budget")
        return {
            "url": validated, "status": status, "content_type": content_type,
            "body": response_body, "trust": "untrusted_external_data",
            "instructions_authorized": False, "requests_used": self.requests_used,
            "request_budget": self.policy.max_requests,
            "resolved_addresses": list(addresses), "connected_address": connected_address,
            "destination_verified": destination_verified,
        }


__all__ = ["ExternalAccessBlocked", "ExternalPolicy", "ExternalSession", "resolve_public_addresses", "validate_external_url"]
