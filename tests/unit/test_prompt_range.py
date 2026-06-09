"""Per-window prompt range: deterministic, per-env, half-open, tiny-env no-op."""

from reliquary.shared.prompt_range import window_prompt_range


def test_deterministic_same_inputs():
    a = window_prompt_range("deadbeef", "openmathinstruct", 880_000, 5000)
    b = window_prompt_range("deadbeef", "openmathinstruct", 880_000, 5000)
    assert a == b


def test_window_is_size_wide_and_in_bounds():
    lo, hi = window_prompt_range("deadbeef", "openmathinstruct", 880_000, 5000)
    assert hi - lo == 5000
    assert 0 <= lo
    assert hi <= 880_000


def test_per_env_diverges():
    # Same seeds, different env names must produce different windows.
    math_los = [
        window_prompt_range(f"s{i}", "openmathinstruct", 880_000, 5000)[0]
        for i in range(50)
    ]
    code_los = [
        window_prompt_range(f"s{i}", "opencode", 880_000, 5000)[0]
        for i in range(50)
    ]
    assert math_los != code_los


def test_randomness_spreads_window():
    los = {
        window_prompt_range(f"seed{i}", "openmathinstruct", 880_000, 5000)[0]
        for i in range(200)
    }
    assert len(los) > 150  # 200 distinct seeds -> mostly distinct windows


def test_tiny_env_is_no_op():
    # universe_n <= size -> whole space eligible (covers test envs)
    assert window_prompt_range("deadbeef", "test", 100, 5000) == (0, 100)
    assert window_prompt_range("deadbeef", "test", 5000, 5000) == (0, 5000)


def test_membership_is_half_open():
    lo, hi = window_prompt_range("deadbeef", "openmathinstruct", 880_000, 5000)
    assert lo in range(lo, hi)
    assert hi not in range(lo, hi)
