from nodus_context import ContextBudget, estimate_tokens, count_tokens


def test_initial_state():
    b = ContextBudget(max_tokens=1000)
    assert b.current_usage == 0
    assert b.remaining == 1000
    assert b.utilization == 0.0
    assert not b.is_warning
    assert not b.is_full


def test_charge_and_release():
    b = ContextBudget(max_tokens=1000)
    b.charge(400)
    assert b.current_usage == 400
    assert b.remaining == 600
    b.release(200)
    assert b.current_usage == 200
    b.release(9999)   # clamp to 0
    assert b.current_usage == 0


def test_utilization():
    b = ContextBudget(max_tokens=1000)
    b.charge(800)
    assert abs(b.utilization - 0.8) < 0.001


def test_is_warning_at_threshold():
    b = ContextBudget(max_tokens=1000, warning_threshold=0.8)
    b.charge(799)
    assert not b.is_warning
    b.charge(1)
    assert b.is_warning


def test_is_full():
    b = ContextBudget(max_tokens=100)
    b.charge(100)
    assert b.is_full
    b.charge(1)
    assert b.is_full   # still full (over limit)


def test_would_overflow():
    b = ContextBudget(max_tokens=100)
    b.charge(90)
    assert not b.would_overflow(10)
    assert b.would_overflow(11)


def test_reset():
    b = ContextBudget(max_tokens=100)
    b.charge(50)
    b.reset()
    assert b.current_usage == 0


# ── estimate_tokens ────────────────────────────────────────────────────────────

def test_estimate_none():
    assert estimate_tokens(None) == 0


def test_estimate_string():
    assert estimate_tokens("hello world") > 0


def test_estimate_empty_string():
    assert estimate_tokens("") == 0


def test_estimate_list():
    result = estimate_tokens(["hello", "world"])
    assert result > 0


def test_count_tokens_falls_back_to_estimate():
    # Without tiktoken installed, should still return something > 0
    result = count_tokens("hello world")
    assert result > 0
