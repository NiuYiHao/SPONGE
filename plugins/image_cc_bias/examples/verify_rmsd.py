"""Verify both generated RMSD bundles against the archived reference outputs."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct

BASE = Path(__file__).resolve().parent


def verify_case(system, out_root):
    name = system.lower()
    directory = out_root / name
    stem = directory / f"{name}_rmsd"
    png = stem.with_suffix(".png").read_bytes()
    if png[:8] != b"\x89PNG\r\n\x1a\n" or len(png) < 1000:
        raise ValueError(f"{system}: valid PNG output required")
    width, height = struct.unpack(">II", png[16:24])
    if min(width, height) < 100:
        raise ValueError(f"{system}: PNG dimensions are too small")
    if not stem.with_suffix(".pdf").read_bytes().startswith(b"%PDF-"):
        raise ValueError(f"{system}: valid PDF output required")
    if "<svg" not in stem.with_suffix(".svg").read_text():
        raise ValueError(f"{system}: valid SVG output required")
    with stem.with_suffix(".csv").open(newline="") as handle:
        actual_rows = list(csv.DictReader(handle))
    with (BASE / name / "rmsd.csv").open(newline="") as handle:
        expected_rows = list(csv.DictReader(handle))
    if actual_rows != expected_rows:
        raise ValueError(f"{system}: per-target CSV differs from the archived values")
    actual = json.loads(stem.with_suffix(".json").read_text())
    expected = json.loads((BASE / name / "rmsd.json").read_text())
    if set(actual) != set(expected):
        raise ValueError(f"{system}: numerical summary fields differ")
    for key, value in expected.items():
        observed = actual[key]
        if isinstance(value, float):
            equal = isinstance(observed, (int, float)) and math.isclose(
                observed, value, rel_tol=1e-12, abs_tol=1e-12)
        else:
            equal = observed == value
        if not equal:
            raise ValueError(f"{system}: summary field {key} differs")
    digest = hashlib.sha256((BASE / "data" / "rmsd_records.csv").read_bytes()).hexdigest()
    if digest != actual["source_rmsd_records_sha256"]:
        raise ValueError(f"{system}: source-data checksum differs")
    result = dict(status="PASS", system=system, n_targets=len(actual_rows),
                  selected_target=actual["selected_target"],
                  baseline_rmsd_A=actual["selected_baseline_rmsd_A"],
                  flexible_rmsd_A=actual["selected_flexible_rmsd_A"],
                  png_dimensions=[width, height], source_sha256=digest)
    print(f"PASS {system}: {len(actual_rows)} targets; case {result['selected_target']} "
          f"RMSD {result['baseline_rmsd_A']:.6f} → {result['flexible_rmsd_A']:.6f} Å")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    results = [verify_case(system, args.out_root) for system in ("1AKE", "FUS")]
    path = args.out_root / "verification.json"
    path.write_text(json.dumps({"status": "PASS", "cases": results}, indent=2) + "\n")
    print(f"Saved verification: {path}")


if __name__ == "__main__":
    main()
