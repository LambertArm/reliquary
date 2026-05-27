r"""Detect correct->wrong boxed-answer flips used to manufacture reward vectors.

The OMI reward scores the LAST ``\boxed{...}``. A miner can flip a correct
rollout to reward 0 by corrupting the final box (appending an empty/special-token
or unclosed box, or boxing the ground truth earlier then overriding it). This
lets a group land on exactly k=4 / sigma=0.5 to pass the zone filter while
keeping emission. Pure, side-effect-free; called by the batcher before GRAIL.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from reliquary.environment.openmathinstruct import _normalize_answer

# Stop/special tokens that must never appear inside a final answer box.
SPECIAL_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")

_MARKER = r"\boxed{"


@dataclass(frozen=True)
class BoxedSpan:
    content: str
    well_formed: bool


def extract_boxed_spans(text: str) -> list[BoxedSpan]:
    r"""All ``\boxed{...}`` occurrences with a well-formed flag.

    Malformed when unclosed, empty/whitespace, or containing a special token.
    """
    spans: list[BoxedSpan] = []
    i = 0
    while True:
        j = text.find(_MARKER, i)
        if j == -1:
            break
        k = j + len(_MARKER)
        depth = 1
        buf: list[str] = []
        closed = False
        while k < len(text):
            c = text[k]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    closed = True
                    k += 1
                    break
            buf.append(c)
            k += 1
        content = "".join(buf)
        well_formed = (
            closed
            and content.strip() != ""
            and not any(tok in content for tok in SPECIAL_TOKENS)
        )
        spans.append(BoxedSpan(content=content, well_formed=well_formed))
        i = k if k > j + len(_MARKER) else j + len(_MARKER)
    return spans


def is_reward_manipulated(
    reward: float, text: str, ground_truth: str
) -> tuple[bool, Optional[str]]:
    """True when a reward=0 rollout shows a correct->wrong boxed flip.

    Only evaluated for reward < 0.5. Conditions:
      (a) "boxed_gt_earlier": a well-formed span normalizes to the ground truth.
      (b) "malformed_final": the last span is malformed while an earlier
          well-formed span exists.
    """
    if reward is not None and reward >= 0.5:
        return False, None
    gt = _normalize_answer(ground_truth)
    if gt == "":
        return False, None
    spans = extract_boxed_spans(text)
    if not spans:
        return False, None
    well_formed = [s for s in spans if s.well_formed]
    if any(_normalize_answer(s.content) == gt for s in well_formed):
        return True, "boxed_gt_earlier"
    if not spans[-1].well_formed and well_formed:
        return True, "malformed_final"
    return False, None
