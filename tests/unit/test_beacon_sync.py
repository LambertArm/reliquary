"""Tests for deterministic beacon round selection.

Miner and validator must agree on the same drand round for a given window.
The round is derived from the window_start block number, not fetched as 'latest'.
"""

from reliquary.constants import BLOCK_TIME_SECONDS, WINDOW_LENGTH


class TestDeterministicBeaconRound:
    def test_compute_window_randomness_includes_round(self):
        """Window randomness must bind the drand round number to prevent
        a miner from choosing a favorable round."""
        from reliquary.infrastructure.chain import compute_window_randomness

        block_hash = "aa" * 32
        drand_rand = "bb" * 32

        r1 = compute_window_randomness(block_hash, drand_rand, drand_round=100)
        r2 = compute_window_randomness(block_hash, drand_rand, drand_round=101)
        r_no_round = compute_window_randomness(block_hash, drand_rand, drand_round=None)

        assert r1 != r2, "Different rounds must produce different randomness"
        assert r1 != r_no_round, "Providing a round must change the result"

    def test_compute_drand_round_for_window(self):
        """Round selection must be deterministic from window_start and chain params."""
        from reliquary.infrastructure.chain import compute_drand_round_for_window

        genesis_time = 1000
        period = 3

        # Window at block 100: timestamp = 100 * 12 = 1200
        # Expected round = 1 + (1200 - 1000) // 3 = 1 + 66 = 67
        r = compute_drand_round_for_window(100, genesis_time, period)
        assert r == 67

        # Same input = same output (deterministic)
        assert compute_drand_round_for_window(100, genesis_time, period) == 67

        # Different window = different round
        assert compute_drand_round_for_window(130, genesis_time, period) != 67

    def test_compute_drand_round_before_genesis_returns_1(self):
        from reliquary.infrastructure.chain import compute_drand_round_for_window

        # Window at block 10: timestamp = 120, before genesis at 1000
        r = compute_drand_round_for_window(10, 1000, 3)
        assert r == 1  # Clamp to round 1 (the first valid round)

    def test_compute_window_randomness_drand_only(self):
        """v2.3+: block_hash may be None — drand-only seed derivation."""
        from reliquary.infrastructure.chain import compute_window_randomness

        drand_rand = "bb" * 32
        r_no_block = compute_window_randomness(None, drand_rand, drand_round=42)
        r_with_block = compute_window_randomness("aa" * 32, drand_rand, drand_round=42)

        # Both forms valid, but produce different seeds.
        assert isinstance(r_no_block, str) and len(r_no_block) == 64
        assert r_no_block != r_with_block

        # Drand-only is still round-bound.
        r_other_round = compute_window_randomness(None, drand_rand, drand_round=43)
        assert r_no_block != r_other_round

    def test_compute_window_randomness_requires_some_source(self):
        """At least one of block_hash or drand_randomness must be provided."""
        import pytest

        from reliquary.infrastructure.chain import compute_window_randomness

        with pytest.raises(ValueError):
            compute_window_randomness(None, None, drand_round=42)


class TestOrderingRoundSelection:
    def test_ordering_round_publishes_after_window_close(self):
        """Ordering round R must satisfy round_time(R) > window_close + margin."""
        from reliquary.infrastructure.chain import compute_drand_round_for_ordering

        # window_start=100, WINDOW_LENGTH=5 → close at block 105 → ts 1260
        # genesis_time=1000, period=3 → margin=1 → target_ts = 1261
        # round R s.t. round_time(R) > 1261. round_time(R) = 1000 + (R-1)*3.
        # We need 1000 + 3*(R-1) > 1261 → R-1 > 87 → R > 88 → R = 89.
        r = compute_drand_round_for_ordering(
            window_start_block=100, genesis_time=1000, period=3,
        )
        assert r == 89

        # Sanity: round_time(R) is strictly greater than target close+margin.
        round_time = 1000 + (r - 1) * 3
        assert round_time > 1260 + 1

    def test_ordering_round_strictly_later_than_grail_round(self):
        """The ordering round must be strictly later than the GRAIL round
        for the same window — otherwise miners can grind."""
        from reliquary.infrastructure.chain import (
            compute_drand_round_for_ordering,
            compute_drand_round_for_window,
        )

        grail = compute_drand_round_for_window(100, 1000, 3)
        ordering = compute_drand_round_for_ordering(100, 1000, 3)
        assert ordering > grail

    def test_ordering_round_before_genesis_clamps_to_1(self):
        from reliquary.infrastructure.chain import compute_drand_round_for_ordering

        # Window before drand genesis — clamp to round 1.
        r = compute_drand_round_for_ordering(0, 1000, 3, window_length_blocks=1)
        assert r == 1

    def test_ordering_round_deterministic(self):
        """Same inputs → same round, across calls."""
        from reliquary.infrastructure.chain import compute_drand_round_for_ordering

        r1 = compute_drand_round_for_ordering(100, 1000, 3)
        r2 = compute_drand_round_for_ordering(100, 1000, 3)
        assert r1 == r2
