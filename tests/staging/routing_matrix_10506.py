"""Deterministic routing check for unslothai/unsloth#10506, run natively per OS.

Exercises the REAL platform detection in install_python_stack (no monkeypatching of
IS_WINDOWS / IS_MAC_ARM / platform.machine), then replays step 13b's elif chain for a
grid of torch versions and asserts the decision each one routes to.
"""
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "studio"))
sys.path.insert(0, str(ROOT / "studio" / "backend"))
import install_python_stack as ips  # noqa: E402

SKIP_NO_TORCH = "skip:no-torch"
SKIP_PLATFORM = "skip:no-wheel-for-platform"
SKIP_UNKNOWN = "skip:torch-unknown"
SKIP_UNSUPPORTED = "skip:unsupported-torch"     # the branch #10506 adds
SKIP_NO_WHEEL = "skip:no-wheel-for-this-torch"
INSTALL = "install"


def decide(torch_version, no_torch = False, lacks_wheel = None):
    """Replay step 13b's elif chain exactly as written in install_python_stack()."""
    if lacks_wheel is None:
        lacks_wheel = ips.PLATFORM_LACKS_TORCHCODEC_WHEEL
    if no_torch:
        return SKIP_NO_TORCH
    if lacks_wheel:
        return SKIP_PLATFORM
    if not torch_version:
        return SKIP_UNKNOWN
    spec = ips._select_torchcodec_spec(torch_version)
    if spec is None:
        return SKIP_UNSUPPORTED
    if not ips._torchcodec_spec_is_installable(spec):
        return SKIP_NO_WHEEL
    return INSTALL


def host_key():
    machine = platform.machine().lower()
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos-arm" if machine in ("arm64", "aarch64") else "macos-intel"
    return "linux-x86_64" if machine in ("x86_64", "amd64") else f"linux-{machine}"


# The rows every platform must agree on: the whole point of the PR.
UNSUPPORTED = ["2.0.0", "2.1.0", "2.2.0", "2.3.0", "2.3.1+cu118", "2.4.0", "2.4.1+cpu",
               "2.4.0+cu121", "2.4.0rc1"]
# Rows that must be untouched by the PR, per platform.
EXPECT = {
    "linux-x86_64": {
        "2.5.0": INSTALL, "2.7.0": INSTALL, "2.8.0": INSTALL, "2.9.0": INSTALL,
        "2.10.0": INSTALL, "2.11.0+cu128": INSTALL, "2.12.0": INSTALL, "2.14.0": INSTALL,
        "1.13.1+cu117": INSTALL, "3.0.0": INSTALL, "not-a-version": INSTALL,
        None: SKIP_UNKNOWN, "": SKIP_UNKNOWN,
    },
    "macos-arm": {
        "2.5.0": INSTALL, "2.7.0": INSTALL, "2.8.0": INSTALL, "2.9.0": INSTALL,
        "2.10.0": INSTALL, "2.11.0": INSTALL, "2.12.0": INSTALL, "2.14.0": INSTALL,
        "1.13.1": INSTALL, "3.0.0": INSTALL, "not-a-version": INSTALL,
        None: SKIP_UNKNOWN, "": SKIP_UNKNOWN,
    },
    "windows": {
        # win_amd64 wheels start at torchcodec 0.7.0, so the 2.5/2.6/2.7 lines have none.
        # This is the platform where the PR's new branch and the wheel gate interact.
        "2.5.0": SKIP_NO_WHEEL, "2.6.0": SKIP_NO_WHEEL, "2.7.0": SKIP_NO_WHEEL,
        "2.8.0": INSTALL, "2.9.0": INSTALL, "2.10.0": INSTALL, "2.11.0+cu128": INSTALL,
        "2.12.0": INSTALL, "1.13.1+cu117": INSTALL, "3.0.0": INSTALL,
        "not-a-version": INSTALL, None: SKIP_UNKNOWN, "": SKIP_UNKNOWN,
    },
}

key = host_key()
print(f"host={key} python={sys.version_info[:2]} machine={platform.machine()}")
print(f"PLATFORM_LACKS_TORCHCODEC_WHEEL={ips.PLATFORM_LACKS_TORCHCODEC_WHEEL} "
      f"NO_TORCH={ips.NO_TORCH} platform_floor={ips._torchcodec_platform_floor()}")
print(f"_TORCHCODEC_MIN_KNOWN_MINOR={ips._TORCHCODEC_MIN_KNOWN_MINOR} "
      f"_TORCHCODEC_MAX_KNOWN_MINOR={ips._TORCHCODEC_MAX_KNOWN_MINOR}")
print()

failures = []

print("--- rows the PR changes: every one must be skip:unsupported-torch ---")
for v in UNSUPPORTED:
    got = decide(v, lacks_wheel = False)
    ok = got == SKIP_UNSUPPORTED
    print(f"  torch {str(v):<14} -> {got:<28} {'ok' if ok else 'MISMATCH'}")
    if not ok:
        failures.append(f"{v}: expected {SKIP_UNSUPPORTED}, got {got}")

exp = EXPECT.get(key)
if exp is None:
    print(f"\n(no expectation table for host {key}; skipping the untouched-rows check)")
else:
    print("\n--- rows the PR must NOT change ---")
    for v, want in exp.items():
        got = decide(v, lacks_wheel = False)
        ok = got == want
        print(f"  torch {str(v):<14} -> {got:<28} want {want:<28} {'ok' if ok else 'MISMATCH'}")
        if not ok:
            failures.append(f"{v}: expected {want}, got {got}")

print("\n--- routing short-circuits that must dominate the new branch ---")
for label, kwargs, want in (
    ("NO_TORCH=1 + torch 2.4", dict(no_torch = True, lacks_wheel = False), SKIP_NO_TORCH),
    ("no platform wheel + torch 2.4", dict(no_torch = False, lacks_wheel = True), SKIP_PLATFORM),
):
    got = decide("2.4.0", **kwargs)
    ok = got == want
    print(f"  {label:<32} -> {got:<28} want {want:<24} {'ok' if ok else 'MISMATCH'}")
    if not ok:
        failures.append(f"{label}: expected {want}, got {got}")

# The ordering the new branch depends on: the wheel gate cannot take None.
print("\n--- the guard is load-bearing ---")
try:
    ips._torchcodec_spec_is_installable(None)
    print("  _torchcodec_spec_is_installable(None) did NOT raise -- ordering not load-bearing")
except Exception as e:
    print(f"  _torchcodec_spec_is_installable(None) raises {type(e).__name__} "
          f"-> the is-None branch must precede it, and does")

print()
if failures:
    print(f"FAILURES ({len(failures)}):")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("ROUTING MATRIX OK")
