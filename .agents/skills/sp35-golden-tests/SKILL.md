---
name: sp35-golden-tests
description: Run and report the SP:35 Chapter 7 golden tests (sub-component scores, Yamuna-bridge BHI, MPI examples) plus the independent reference check. Use after any change to kb/ or sp35/, before merging, and at every gate.
---

# SP:35 golden tests

1. Run the independent oracle (no project imports):
   `uv run --with pyyaml python tools/sp35_reference_check.py`
   Every line must end in `PASS`. It also checks that `kb/sp35_2024/components.yaml` and `mpi.yaml` carry the same weights as the book.
2. Run the engine tests: `uv run --group dev pytest tests/golden tests/property tests/unit/test_engine.py`.
3. Report a table: case, expected (book), tolerance, oracle value, engine value, pass/fail.
4. If anything fails:
   - Do not edit expected values.
   - Find the code or KB error first.
   - If the book itself is wrong, add an entry to `docs/plan/SP35_CH7_INTERPRETATIONS.md` and ask the user for sign-off before changing behaviour.
