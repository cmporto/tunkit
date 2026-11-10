#!/usr/bin/env python3
"""
Golden value checking runner.

Purpose: check stability between changes in code and, in the future,
be a complete tester.
Comparison is bitwise exact by default, but it can be changed.

Layout:

    tests/
      run_golden.py            <- you're here
      <test_folder>/
        case.py                <- prepare the run(API) output dict
        <data files>
        golden.json            <- written by --save

Any subfolder containing a case.py is a test.

Usage:
    python run_golden.py                    check every test
    python run_golden.py --only             <folder>
    python run_golden.py --save             write/overwrite golden.json

"""



import os

# Must be set before numpy is imported. Multithreaded BLAS may
# make the tests flake... ;)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


import argparse
import importlib.util
import json
import sys
import traceback

import numpy as np

# Agg makes matplotlib plot in a non-interactive backend
# stops windows opening or other shenaningans
import matplotlib
matplotlib.use("Agg")


HERE = os.path.dirname(os.path.abspath(__file__))

# 0.A single import
# any change in the structure of the class is made here and propagated downward
sys.path.insert(0, os.path.dirname(HERE))
from tunkit import WKB_Tunneling_Analysis as API


# 1 comparison
# 1.A building the arrays
def _numeric(x):
    """Return x as a float array, or None if it is not numeric data."""
    if isinstance(x, (str, bool, type(None))):
        return None
    try:
        return np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        return None

# 1.B flexible comparison
def compare(got, expected, tol=None):
    """
    Return (ok, detail). tol is None for exact comparison, or a relative
    tolerance if specified in the case.py file.
    """
    a, b = _numeric(got), _numeric(expected)

    if a is None or b is None:
        ok = got == expected
        return bool(ok), "" if ok else f"{expected!r} -> {got!r}"

    if a.shape != b.shape:
        return False, f"shape {b.shape} -> {a.shape}"

    if tol is None:
        if np.array_equal(a, b, equal_nan=True):
            return True, ""
    else:
        if np.all(np.isclose(a, b, rtol=0, atol=tol, equal_nan=True)): #rtol=tol,atol=0.0
            return True, ""


    # If there are any mismatches, locate the worst offender (NaN prority)
    with np.errstate(invalid="ignore"):
        diff = np.abs(a - b)
    diff = np.where(np.isnan(diff), np.inf, diff)   # a NaN mismatch is worst
    flat = int(np.argmax(diff))
    where = np.unravel_index(flat, a.shape) if a.ndim > 1 else flat
    worst = diff.flat[flat]
    worst_s = "NaN mismatch" if not np.isfinite(worst) else f"{worst:.3e}"
    return False, (f"max |delta| = {worst_s} at {where}: "
                   f"{b.flat[flat]!r} -> {a.flat[flat]!r}")



# 2. preparing results
# 2.A validations
def validate(results):
    """
    Catch values that would be saved but are empty.
    an empty array would pass the test but check nothing and
    hide mistakes.
    """
    problems = []
    for key in sorted(results):
        val = results[key]
        if val is None:
            problems.append((key, "is None"))
            continue
        arr = _numeric(val)
        if arr is None:
            continue
        if arr.size == 0:
            problems.append((key, "is empty"))
        elif np.all(np.isnan(arr)):
            problems.append((key, "is all NaN"))
    return problems



# 2.B transforming to lists
def to_jsonable(obj):
    """json writes floats with repr(), which round-trips float64 exactly."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


# 3. discovery and execution
# 3.A get the folders
def find_cases(only=None):
    cases = []
    for name in sorted(os.listdir(HERE)):
        folder = os.path.join(HERE, name)
        if not os.path.isdir(folder):
            continue
        if not os.path.isfile(os.path.join(folder, "case.py")):
            continue
        if only and name != only:
            continue
        cases.append((name, folder))
    return cases


# 3.B loading the folders
def load_case(name, folder):
    spec = importlib.util.spec_from_file_location(
        f"case_{name}", os.path.join(folder, "case.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute(module, folder):
    """Run the case with folder as the working directory."""
    cwd = os.getcwd()
    os.chdir(folder)
    try:
        return module.run(API)
    finally:
        os.chdir(cwd)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true",
                    help="write golden.json instead of comparing")
    ap.add_argument("--only", default=None, help="run a single test folder")
    args = ap.parse_args()

    cases = find_cases(args.only)
    if not cases:
        print("No test folders found (a test folder is one containing case.py).")
        return 1

    failed = []

    for name, folder in cases:
        try:
            module = load_case(name, folder)
            results = execute(module, folder)
        except Exception:
            print(f"{name}   ERROR while running")
            traceback.print_exc()
            failed.append(name)
            continue

        gpath = os.path.join(folder, "golden.json")

        if args.save:
            really_save = input("Do you really want to save and overwrite previous values? [y,n] \n")
            if really_save.lower() == 'y':
                problems = validate(results)
                if problems and not args.force:
                    print(f"{name}   NOT SAVED -- suspicious values:")
                    for key, why in problems:
                        print(f"    {key:<28} {why}")
                    print("    fix the case, or re-run with --force if intended")
                    failed.append(name)
                    continue
                with open(gpath, "w") as fh:
                    json.dump(to_jsonable(results), fh, indent=1, sort_keys=True)
                print(f"{name}   saved {len(results)} quantities")
                continue
            else:
                print('Skipping saving...')
                args.save = False

        if not os.path.isfile(gpath):
            print(f"{name}   no golden.json -- run with --save first")
            failed.append(name)
            continue

        with open(gpath) as fh:
            golden = json.load(fh)

        tolerances = getattr(module, "TOLERANCES", {})
        problems = []

        for key in sorted(golden):
            if key not in results:
                problems.append((key, "MISSING from result"))
                continue
            ok, detail = compare(results[key], golden[key], tolerances.get(key))
            if not ok:
                problems.append((key, detail))

        if problems:
            print(f"{name}   FAILED")
            for key, detail in problems:
                print(f"    {key:<28} {detail}")
            failed.append(name)
        else:
            print(f"{name}   Passed!  ({len(golden)} quantities)")

        for key in sorted(set(results) - set(golden)):
            print(f"    {key:<28} new, not in golden.json (run --save to record)")

    if args.save:
        print("\nSaved. Read the git diff before committing.")
        return 0

    if failed:
        print(f"\n{len(failed)} of {len(cases)} failed: {', '.join(failed)}")
        return 1

    # import numpy as np
    # from dataclasses import fields
    # from tunkit import WKB_Tunneling_Analysis
    # from tunkit.path.reaction_path import ReactionPath

    # wkb = WKB_Tunneling_Analysis()
    # wkb.load_cluster_results(json_file='path_from_payload_json/carb_CH3_singlet_projections.json',
    #                          verbose=False)
    # wkb.build_reaction_path(verbose=False)

    # before = {f.name: np.asarray(getattr(wkb, f.name))
    #         for f in fields(ReactionPath) if hasattr(wkb, f.name)}

    # wkb.from_path(wkb.to_path())          # round trip

    # for name, old in before.items():
    #     new = np.asarray(getattr(wkb, name))
    #     if old.shape != new.shape or not np.array_equal(old, new, equal_nan=True):
    #         print(f"CHANGED: {name}  {old.shape} -> {new.shape}")
    # else:
    #     print("round trip clean")

    # print(f"\nAll {len(cases)} passed.")
    # return 0


if __name__ == "__main__":
    sys.exit(main())
