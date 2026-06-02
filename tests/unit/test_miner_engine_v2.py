"""Miner prompt-picking strategy: pull random in-range, skip cooldown."""

import random
from unittest.mock import MagicMock

import pytest

from reliquary.miner.engine import pick_prompt_idx


class FakeEnv:
    def __len__(self):
        return 100


def test_pick_prompt_in_range():
    env = FakeEnv()
    rng = random.Random(42)
    for _ in range(50):
        idx = pick_prompt_idx(env, cooldown_prompts=set(), rng=rng)
        assert 0 <= idx < 100


def test_pick_prompt_skips_cooldown():
    env = FakeEnv()
    rng = random.Random(42)
    cooldown = set(range(0, 95))  # only 5 choices free: 95..99
    for _ in range(20):
        idx = pick_prompt_idx(env, cooldown_prompts=cooldown, rng=rng)
        assert idx not in cooldown


def test_pick_prompt_all_cooldown_raises():
    env = FakeEnv()
    rng = random.Random(42)
    cooldown = set(range(100))
    with pytest.raises(RuntimeError, match="no eligible prompt"):
        pick_prompt_idx(env, cooldown_prompts=cooldown, rng=rng)


def test_engine_default_max_new_tokens_is_protocol_cap():
    """The env-var override is removed; max_new_tokens is the protocol cap."""
    import inspect
    from reliquary.constants import MAX_NEW_TOKENS_PROTOCOL_CAP
    from reliquary.miner.engine import MiningEngine

    sig = inspect.signature(MiningEngine.__init__)
    default = sig.parameters["max_new_tokens"].default
    assert default == MAX_NEW_TOKENS_PROTOCOL_CAP

    # Belt and suspenders: source must not reference the env var anywhere
    # in the engine module — catches a regression that re-introduces the
    # `int(os.environ.get("RELIQUARY_MAX_NEW_TOKENS", ...))` default.
    src = inspect.getsource(MiningEngine.__init__)
    assert "RELIQUARY_MAX_NEW_TOKENS" not in src


def test_build_rollout_submission_uses_placeholder_for_authoritative_reward_env():
    from reliquary.miner.engine import MiningEngine

    class _PrivateRewardEnv:
        name = "opencodeinstruct"
        validator_authoritative_reward = True
        compute_reward = MagicMock(side_effect=AssertionError("must not score locally"))

    eng = object.__new__(MiningEngine)
    eng.env = _PrivateRewardEnv()
    eng.tokenizer = MagicMock()
    eng.tokenizer.decode.return_value = "```python\ndef add(a, b): return a + b\n```"
    eng._build_grail_commit = MagicMock(
        return_value={
            "tokens": [10, 11, 12, 13],
            "rollout": {"prompt_length": 2, "completion_length": 2},
        }
    )
    generation = {"tokens": [10, 11, 12, 13], "prompt_length": 2}

    rollout = eng._build_rollout_submission(
        generation, {"prompt": "p"}, "randomness", env=eng.env
    )

    assert rollout.reward == 0.0
    assert rollout.env_name == "opencodeinstruct"
    eng.env.compute_reward.assert_not_called()
