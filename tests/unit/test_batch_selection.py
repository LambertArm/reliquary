"""Drand-anchored batch selection + emission distribution tests."""

from collections import defaultdict
from dataclasses import dataclass

from reliquary.validator.batch_selection import select_batch_and_distribute
from reliquary.validator.cooldown import CooldownMap


@dataclass
class FakeSubmission:
    hotkey: str
    prompt_idx: int
    merkle_root: bytes = b"\x00" * 32


def _sub(hotkey, prompt_idx, merkle_root=None):
    return FakeSubmission(
        hotkey=hotkey,
        prompt_idx=prompt_idx,
        merkle_root=merkle_root or hotkey.encode().ljust(32, b"\x00"),
    )


def _by_prompt(subs):
    out: dict[int, list] = defaultdict(list)
    for s in subs:
        out[s.prompt_idx].append(s)
    return dict(out)


SEED = b"\x42" * 32
OTHER_SEED = b"\x99" * 32


def test_empty_pool_returns_empty():
    cd = CooldownMap(cooldown_windows=50)
    batch, rewards = select_batch_and_distribute(
        submissions_by_prompt={}, b=8, ordering_seed=SEED,
        cooldown_map=cd, current_window=100,
    )
    assert batch == [] and rewards == {}


def test_fills_up_to_b_distinct_prompts():
    cd = CooldownMap(cooldown_windows=50)
    subs = [_sub(f"hk{i}", prompt_idx=i) for i in range(12)]
    batch, rewards = select_batch_and_distribute(
        submissions_by_prompt=_by_prompt(subs), b=8, ordering_seed=SEED,
        cooldown_map=cd, current_window=100, pool=1.0,
    )
    assert len(batch) == 8
    # Each unique prompt has 1 miner → each miner gets 1/8.
    assert abs(sum(rewards.values()) - 1.0) < 1e-9
    assert all(abs(v - 1/8) < 1e-9 for v in rewards.values())


def test_ordering_is_independent_of_arrival():
    """Two different insertion orders into submissions_by_prompt produce
    the same winning prompts (latency-independent ordering)."""
    cd = CooldownMap(cooldown_windows=50)
    prompts = list(range(20))
    subs = [_sub(f"hk{p}", prompt_idx=p) for p in prompts]

    by_prompt_a = _by_prompt(subs)
    by_prompt_b = _by_prompt(list(reversed(subs)))

    batch_a, _ = select_batch_and_distribute(
        submissions_by_prompt=by_prompt_a, b=8, ordering_seed=SEED,
        cooldown_map=cd, current_window=100,
    )
    batch_b, _ = select_batch_and_distribute(
        submissions_by_prompt=by_prompt_b, b=8, ordering_seed=SEED,
        cooldown_map=cd, current_window=100,
    )
    assert {s.prompt_idx for s in batch_a} == {s.prompt_idx for s in batch_b}


def test_ordering_changes_with_seed():
    """Different drand seeds produce different winning sets (in general)."""
    cd = CooldownMap(cooldown_windows=50)
    subs = [_sub(f"hk{p}", prompt_idx=p) for p in range(20)]
    by_prompt = _by_prompt(subs)

    batch_a, _ = select_batch_and_distribute(
        submissions_by_prompt=by_prompt, b=8, ordering_seed=SEED,
        cooldown_map=cd, current_window=100,
    )
    batch_b, _ = select_batch_and_distribute(
        submissions_by_prompt=by_prompt, b=8, ordering_seed=OTHER_SEED,
        cooldown_map=cd, current_window=100,
    )
    # With 20 prompts and a different seed, the top 8 must differ for at
    # least one element with overwhelming probability under SHA256.
    assert {s.prompt_idx for s in batch_a} != {s.prompt_idx for s in batch_b}


def test_cooldown_excludes_prompt_from_winners():
    cd = CooldownMap(cooldown_windows=50)
    cd.record_batched(prompt_idx=42, window=100)
    subs = [_sub("hk", prompt_idx=42), _sub("hk2", prompt_idx=7)]
    batch, rewards = select_batch_and_distribute(
        submissions_by_prompt=_by_prompt(subs), b=8, ordering_seed=SEED,
        cooldown_map=cd, current_window=110,
    )
    assert [s.prompt_idx for s in batch] == [7]
    assert set(rewards) == {"hk2"} and abs(rewards["hk2"] - 1.0) < 1e-9


def test_cooldown_not_mutated_by_select():
    cd = CooldownMap(cooldown_windows=50)
    subs = [_sub("hk", prompt_idx=42)]
    select_batch_and_distribute(
        submissions_by_prompt=_by_prompt(subs), b=8, ordering_seed=SEED,
        cooldown_map=cd, current_window=100,
    )
    assert cd.is_in_cooldown(42, 100) is False


def test_partial_fill_when_all_cooldown_blocked():
    cd = CooldownMap(cooldown_windows=50)
    for idx in (1, 2, 3):
        cd.record_batched(idx, window=100)
    subs = [_sub(f"hk{i}", prompt_idx=i) for i in (1, 2, 3)]
    batch, rewards = select_batch_and_distribute(
        submissions_by_prompt=_by_prompt(subs), b=8, ordering_seed=SEED,
        cooldown_map=cd, current_window=110,
    )
    assert batch == [] and rewards == {}


def test_multi_miner_per_prompt_splits_slot():
    """K miners on the same winning prompt each get slot_share / K."""
    cd = CooldownMap(cooldown_windows=50)
    subs = [
        _sub("alice", prompt_idx=1),
        _sub("bob", prompt_idx=1),
        _sub("carol", prompt_idx=1),
        _sub("dave", prompt_idx=2),
    ]
    batch, rewards = select_batch_and_distribute(
        submissions_by_prompt=_by_prompt(subs), b=8, ordering_seed=SEED,
        cooldown_map=cd, current_window=100, pool=1.0,
    )
    # Two winning prompts, slot_share = 0.5 each.
    # Prompt 1: 3 miners → 0.5/3 = 1/6 each.
    # Prompt 2: 1 miner → 0.5.
    assert abs(rewards["alice"] - 1/6) < 1e-9
    assert abs(rewards["bob"] - 1/6) < 1e-9
    assert abs(rewards["carol"] - 1/6) < 1e-9
    assert abs(rewards["dave"] - 0.5) < 1e-9
    # Training picks one submission per winning prompt.
    assert len(batch) == 2


def test_sybil_neutral_on_same_prompt():
    """Total attacker payout when N sybils target the same prompt equals
    the payout of a single hotkey winning that prompt alone."""
    cd = CooldownMap(cooldown_windows=50)

    # Case A: lone attacker on prompt 1, honest miner on prompt 2.
    lone = [_sub("attacker", prompt_idx=1), _sub("honest", prompt_idx=2)]
    _, rewards_a = select_batch_and_distribute(
        submissions_by_prompt=_by_prompt(lone), b=8, ordering_seed=SEED,
        cooldown_map=cd, current_window=100, pool=1.0,
    )

    # Case B: 5 attacker sybils on prompt 1, honest miner on prompt 2.
    sybils = [_sub(f"sybil{i}", prompt_idx=1) for i in range(5)]
    sybils.append(_sub("honest", prompt_idx=2))
    _, rewards_b = select_batch_and_distribute(
        submissions_by_prompt=_by_prompt(sybils), b=8, ordering_seed=SEED,
        cooldown_map=cd, current_window=100, pool=1.0,
    )

    attacker_a = rewards_a["attacker"]
    attacker_b_total = sum(v for k, v in rewards_b.items() if k.startswith("sybil"))
    assert abs(attacker_a - attacker_b_total) < 1e-9

    # And the honest miner is unaffected by the sybil flood on the other prompt.
    assert abs(rewards_a["honest"] - rewards_b["honest"]) < 1e-9


def test_training_pick_deterministic_per_seed():
    """Same seed + same submissions → same training picks across calls."""
    cd = CooldownMap(cooldown_windows=50)
    subs = [_sub(f"hk{i}", prompt_idx=1) for i in range(5)]
    by_prompt = _by_prompt(subs)

    batch_1, _ = select_batch_and_distribute(
        submissions_by_prompt=by_prompt, b=8, ordering_seed=SEED,
        cooldown_map=cd, current_window=100,
    )
    batch_2, _ = select_batch_and_distribute(
        submissions_by_prompt=by_prompt, b=8, ordering_seed=SEED,
        cooldown_map=cd, current_window=100,
    )
    assert [s.hotkey for s in batch_1] == [s.hotkey for s in batch_2]


def test_training_pick_canonical_across_input_order():
    """Same seed + same submissions in DIFFERENT in-memory order → same pick.
    Required for multi-validator consensus where _valid order differs."""
    cd = CooldownMap(cooldown_windows=50)
    subs = [_sub(f"hk{i}", prompt_idx=1) for i in range(5)]

    by_prompt_a = {1: list(subs)}
    by_prompt_b = {1: list(reversed(subs))}

    batch_a, _ = select_batch_and_distribute(
        submissions_by_prompt=by_prompt_a, b=8, ordering_seed=SEED,
        cooldown_map=cd, current_window=100,
    )
    batch_b, _ = select_batch_and_distribute(
        submissions_by_prompt=by_prompt_b, b=8, ordering_seed=SEED,
        cooldown_map=cd, current_window=100,
    )
    assert [s.hotkey for s in batch_a] == [s.hotkey for s in batch_b]


def test_pool_scaling():
    """Pool scales rewards linearly."""
    cd = CooldownMap(cooldown_windows=50)
    subs = [_sub(f"hk{i}", prompt_idx=i) for i in range(4)]
    by_prompt = _by_prompt(subs)

    _, rewards_1 = select_batch_and_distribute(
        submissions_by_prompt=by_prompt, b=8, ordering_seed=SEED,
        cooldown_map=cd, current_window=100, pool=1.0,
    )
    _, rewards_100 = select_batch_and_distribute(
        submissions_by_prompt=by_prompt, b=8, ordering_seed=SEED,
        cooldown_map=cd, current_window=100, pool=100.0,
    )
    for hk in rewards_1:
        assert abs(rewards_100[hk] - rewards_1[hk] * 100.0) < 1e-7
