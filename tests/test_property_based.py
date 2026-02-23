"""Property-based tests using hypothesis for edge case discovery."""

import typing

from hypothesis import given, settings as h_settings, strategies as st

from src.agent.guardrails import audit_input, detect_responsible_gaming
from src.agent.streaming_pii import StreamingPIIRedactor
from src.api.pii_redaction import redact_pii


# Test 1: PII redactor never crashes on arbitrary input
@given(st.text(min_size=0, max_size=1000))
@h_settings(max_examples=200)
def test_pii_redactor_never_crashes(text):
    result = redact_pii(text)
    assert isinstance(result, str)


# Test 2: Guardrails never crash on arbitrary input
@given(st.text(min_size=0, max_size=500))
@h_settings(max_examples=200)
def test_guardrails_never_crash(text):
    safe = audit_input(text)
    assert isinstance(safe, bool)
    rg = detect_responsible_gaming(text)
    assert isinstance(rg, bool)


# Test 3: Streaming PII redactor handles arbitrary chunks
@given(st.lists(st.text(min_size=0, max_size=100), min_size=0, max_size=20))
@h_settings(max_examples=100)
def test_streaming_redactor_never_crashes(chunks):
    redactor = StreamingPIIRedactor()
    for chunk in chunks:
        list(redactor.feed(chunk))  # consume iterator
    list(redactor.flush())  # consume remaining


# Test 4: Router output Literal types are exhaustive
def test_router_output_query_types_exhaustive():
    from src.agent.state import RouterOutput

    literal_args = typing.get_args(
        RouterOutput.model_fields["query_type"].annotation
    )
    assert len(literal_args) == 7  # property_qa, hours_schedule, greeting, off_topic, gambling_advice, action_request, ambiguous


# Test 5: IPv6 normalization in rate limiter
def test_rate_limit_normalize_ip_ipv6():
    from src.api.middleware import RateLimitMiddleware

    # Bracketed IPv6
    assert RateLimitMiddleware._normalize_ip("[::1]") == "::1"
    # IPv4-mapped IPv6
    assert RateLimitMiddleware._normalize_ip("::ffff:192.168.1.1") == "192.168.1.1"
    # Plain IPv4 passthrough
    assert RateLimitMiddleware._normalize_ip("10.0.0.1") == "10.0.0.1"
    # Full IPv6 passthrough
    assert RateLimitMiddleware._normalize_ip("2001:db8::1") == "2001:db8::1"
    # Whitespace stripping
    assert RateLimitMiddleware._normalize_ip("  10.0.0.1  ") == "10.0.0.1"
    # Bracketed IPv6 with zone ID
    assert RateLimitMiddleware._normalize_ip("[::1%eth0]") == "::1%eth0"
