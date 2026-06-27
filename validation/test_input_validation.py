"""Fast, weights-free check of Krea2Pipeline.generate input validation.

generate() validates its arguments before touching the model, so we can drive it on a
bare instance (object.__new__, no __init__/weights) and assert the guards fire. Suitable
for CI on any machine with mlx + mflux importable — no model download, runs in <1s.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root → import krea2

from krea2.pipeline import Krea2Pipeline

REJECT = [   # (prompt, kwargs) that must raise ValueError
    ("", {}),                       # empty prompt
    ("   ", {}),                    # whitespace-only prompt
    ("ok", {"width": 1000}),        # not a multiple of 16
    ("ok", {"width": 128}),         # below the 256 floor
    ("ok", {"height": 4096}),       # above the 2048 ceiling
    ("ok", {"steps": 0}),           # steps below 1
    ("ok", {"steps": 99}),          # steps above 50
    ("ok", {"num_images": 0}),      # too few
    ("ok", {"num_images": 9}),      # too many
    ("ok", {"seed": -1}),           # negative seed (mx.random.seed wants non-negative)
    ("ok", {"width": None}),        # non-numeric → uniform ValueError, not TypeError
]


def main() -> int:
    p = object.__new__(Krea2Pipeline)   # skip __init__ — validation runs before any model access
    failures = []

    for prompt, kw in REJECT:
        try:
            p.generate(prompt, **kw)
            failures.append(f"expected ValueError for prompt={prompt!r} kw={kw}")
        except ValueError:
            pass
        except Exception as e:  # any other exception means validation didn't gate it
            failures.append(f"wrong exception {type(e).__name__} for prompt={prompt!r} kw={kw}: {e}")

    # a valid call must pass validation (then fail later at the model step, which we don't reach here)
    try:
        p.generate("a fox", width=1024, height=768, steps=8, num_images=2)
        failures.append("valid call unexpectedly returned (no model loaded)")
    except ValueError as e:
        failures.append(f"valid input wrongly rejected: {e}")
    except AttributeError:
        pass  # expected: passed validation, then hit the missing self.transformer

    if failures:
        print("FAIL:\n  " + "\n  ".join(failures))
        return 1
    print(f"OK: {len(REJECT)} invalid inputs rejected, valid input passed validation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
