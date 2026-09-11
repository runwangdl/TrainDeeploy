# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import os
import re
from pathlib import Path
from typing import Any, Dict, List

import coloredlogs
import pytest

from Deeploy.Logging import DEFAULT_FMT
from Deeploy.Logging import DEFAULT_LOGGER as log

# Accumulates per-test runtime cycles for the end-of-session perf summary.
# Populated by pytest_runtest_logreport (runs on the xdist master, so this
# is single-process even with parallel workers).
_PERF_RESULTS: List[Dict[str, Any]] = []

# Inference harness format
_RUNTIME_CYCLES_RE = re.compile(r"Runtime:\s*(\d+)\s*cycles")
# Training harness format: train + optimizer step cycles + (optional) total bytes
# of the trainable weight tensors. NOTE: a logical sum over the weight slots of
# DeeployNetwork_inputs[], independent of which memory level those tensors
# actually live in -- it is a parameter-count cross-check, not a footprint.
_BENCH_RE = re.compile(r"BENCH\s+train_cycles=(\d+)(?:\s+opt_cycles=(\d+))?(?:\s+trainable_bytes=(\d+))?")

# ---------------------------------------------------------------------------
# Known failures
#
# These run in full -- they are compiled, simulated and numerically compared --
# but their failure is recorded as expected. strict=True means an unexpected
# PASS fails the run, so an entry cannot quietly rot here once the underlying
# problem is fixed: CI will tell you to delete it.
#
# None of these ever passed. They only became visible when the loss comparison
# started running at all (it had been dead code: the guard `pi_core_id() != 0`
# never holds on GAP9, whose cluster controller reports core id 8). Marking them
# records a pre-existing state, it does not accept a regression.
#
# Each entry is (test-function substring, parameter-id substring, reason).
# ---------------------------------------------------------------------------
_LORA_FIXTURE_REASON = ("fixture references are wrong, not the deployment: CCT_LoRA_R1 and "
                        "CCT_QLORA_FT ship byte-identical inputs.npz and outputs.npz "
                        "(md5 2d0875bc../a01d0bba..) although their graphs differ -- 387 nodes with "
                        "no quantisation vs 389 nodes with 14 Dequant. The quantised model is being "
                        "checked against references generated from the float one; the measured error "
                        "is 0.12-0.50, which no defensible tolerance covers. Fix is to regenerate "
                        "the references from each graph.")

_RESNET8_PROMOTE_REASON = ("PromoteTensorsToL2 shifts ResNet-8's FP32 accumulation order: the step-0 "
                           "forward alone is off by 8.7e-4 against a 1e-3 tolerance, where plain L3 is "
                           "bit-exact, and steps 1-3 drift to 0.005-0.025. Reproduced 3/3 with the same "
                           "3-of-4 count. MobileNetV1 and CCT-2 take the same promote path and stay "
                           "within 1e-4, so this looks specific rather than inherent. Not on the "
                           "critical path -- ResNet-8's fastest verified configuration is on-chip L2 "
                           "(116.2 ms), which is both faster than promote and numerically clean.")

_KNOWN_FAILURES = [
    ("test_gap9_tiled_training_l2_singlebuffer", "CCT_QLORA_FT", _LORA_FIXTURE_REASON),
    ("test_gap9_tiled_training_l3_singlebuffer", "CCT_QLORA_FT", _LORA_FIXTURE_REASON),
    ("test_gap9_tiled_training_l3_singlebuffer", "CCT_LoRA_R1", _LORA_FIXTURE_REASON),
    ("test_gap9_tiled_training_promote_l3_singlebuffer", "ResNet8", _RESNET8_PROMOTE_REASON),
    ("test_gap9_tiled_training_promote_l3_doublebuffer", "ResNet8", _RESNET8_PROMOTE_REASON),
]


def pytest_collection_modifyitems(config, items):
    """Attach xfail(strict=True) to the known failures listed above."""
    for item in items:
        for func_part, param_part, reason in _KNOWN_FAILURES:
            if func_part in item.nodeid and param_part in item.nodeid:
                item.add_marker(pytest.mark.xfail(strict = True, reason = reason))
                break


def pytest_addoption(parser: pytest.Parser) -> None:
    """Native PyTest hook: add custom command-line options for Deeploy tests."""
    parser.addoption(
        "--skipgen",
        action = "store_true",
        default = False,
        help = "Skip network generation step",
    )
    parser.addoption(
        "--skipsim",
        action = "store_true",
        default = False,
        help = "Skip simulation step (only generate and build)",
    )
    parser.addoption(
        "--profile-untiled",
        action = "store_true",
        default = False,
        help = "Enable profiling for untiled Siracusa runs",
    )
    parser.addoption(
        "--toolchain",
        action = "store",
        default = "LLVM",
        help = "Compiler toolchain to use (LLVM or GCC)",
    )
    parser.addoption(
        "--toolchain-install-dir",
        action = "store",
        default = os.environ.get("LLVM_INSTALL_DIR"),
        help = "Path to toolchain installation directory",
    )
    parser.addoption(
        "--cmake-args",
        action = "append",
        default = [],
        help = "Additional CMake arguments (can be used multiple times)",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Native PyTest hook: configure pytest for Deeploy tests."""
    # Register custom markers
    config.addinivalue_line("markers", "generic: mark test as a Generic platform test")
    config.addinivalue_line("markers", "cortexm: mark test as a Cortex-M (QEMU-ARM) platform test")
    config.addinivalue_line("markers", "mempool: mark test as a MemPool platform test")
    config.addinivalue_line("markers", "chimera: mark test as a Chimera platform test")
    config.addinivalue_line("markers", "softhier: mark test as a SoftHier platform test")
    config.addinivalue_line("markers", "snitch: mark test as a Snitch platform test")
    config.addinivalue_line("markers", "snitch_tiled: mark test as a Snitch platform test (tiled)")
    config.addinivalue_line("markers", "siracusa: mark test as a Siracusa platform test (untiled)")
    config.addinivalue_line("markers", "siracusa_tiled: mark test as a Siracusa platform test (tiled)")
    config.addinivalue_line("markers",
                            "siracusa_neureka_tiled: mark test as a Siracusa + Neureka platform test (tiled)")
    config.addinivalue_line("markers", "gap9: mark test as a GAP9 platform test")
    config.addinivalue_line("markers", "gap9_tiled: mark test as a GAP9 platform test (tiled)")
    config.addinivalue_line("markers", "kernels: mark test as a kernel test (individual operators)")
    config.addinivalue_line("markers", "models: mark test as a model test (full networks)")
    config.addinivalue_line("markers", "training: mark test as a training-pipeline test (forward + backward + SGD)")
    config.addinivalue_line(
        "markers", "train_kernel: mark test as a training-related single-op kernel test (grad / loss / optimizer)")
    config.addinivalue_line("markers", "singlebuffer: mark test as single-buffer configuration")
    config.addinivalue_line("markers", "doublebuffer: mark test as double-buffer configuration")
    config.addinivalue_line("markers", "l2: mark test as L2 default memory level")
    config.addinivalue_line("markers", "l3: mark test as L3 default memory level")
    config.addinivalue_line(
        "markers", "recompute: mark test as replaying a gradient-checkpointing schedule "
        "(activations regenerated before the backward pass instead of held live)")
    config.addinivalue_line("markers", "wmem: mark test as using Neureka weight memory")
    config.addinivalue_line("markers", "promote: mark test as exercising the L3->L2 tensor promotion pass")
    config.addinivalue_line("markers", "dma: mark test as DMA test")
    config.addinivalue_line(
        "markers",
        "deeploy_internal: mark test as internal Deeploy test (state serialization, extensions, transformations)")

    # Configure logging based on verbosity
    verbosity = config.option.verbose
    if verbosity >= 3:
        coloredlogs.install(level = 'DEBUG', logger = log, fmt = DEFAULT_FMT)
    elif verbosity >= 2:
        coloredlogs.install(level = 'INFO', logger = log, fmt = DEFAULT_FMT)
    else:
        coloredlogs.install(level = 'WARNING', logger = log, fmt = DEFAULT_FMT)


@pytest.fixture(scope = "session")
def deeploy_test_dir():
    """Return the DeeployTest directory path."""
    return Path(__file__).parent


@pytest.fixture(scope = "session")
def tests_dir(deeploy_test_dir):
    """Return the Tests directory path."""
    return deeploy_test_dir / "Tests"


@pytest.fixture(scope = "session")
def toolchain_dir(request):
    """Return the toolchain installation directory."""
    toolchain_install = request.config.getoption("--toolchain-install-dir")
    if toolchain_install is None:
        pytest.skip(reason = "LLVM_INSTALL_DIR not set")
    return toolchain_install


@pytest.fixture(scope = "session", autouse = True)
def ccache_dir():
    """Setup and return ccache directory."""
    # Use existing CCACHE_DIR if already set
    if "CCACHE_DIR" in os.environ:
        return Path(os.environ["CCACHE_DIR"])

    # Fall back to /app/.ccache if it exists (for CI containers)
    ccache_path = Path("/app/.ccache")
    if ccache_path.exists():
        os.environ["CCACHE_DIR"] = str(ccache_path)
        return ccache_path

    return None


@pytest.fixture
def skipgen(request):
    """Return whether to skip network generation."""
    return request.config.getoption("--skipgen")


@pytest.fixture
def skipsim(request):
    """Return whether to skip simulation."""
    return request.config.getoption("--skipsim")


@pytest.fixture
def profile_untiled(request):
    """Return whether untiled profiling is enabled."""
    return request.config.getoption("--profile-untiled")


@pytest.fixture
def toolchain(request):
    """Return the toolchain to use."""
    return request.config.getoption("--toolchain")


@pytest.fixture
def cmake_args(request):
    """Return additional CMake arguments."""
    return request.config.getoption("--cmake-args")


# ---------------------------------------------------------------------------
# Performance summary hooks
#
# pytest_runtest_logreport runs on the xdist master for every worker's report,
# so we can collect per-test runtime cycles in a single process. The Markdown
# summary is emitted to GITHUB_STEP_SUMMARY at session end so PR check pages
# show a perf table inline.
# ---------------------------------------------------------------------------


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Scrape runtime cycles from each test's captured stdout/stderr.

    Two harness formats are recognised:
      * inference  : ``Runtime: N cycles``
      * training   : ``BENCH train_cycles=N opt_cycles=N trainable_bytes=N``
    """
    if report.when != "call":
        return
    if report.outcome not in ("passed", "failed"):
        return

    blob = "\n".join(filter(None, [
        getattr(report, "capstdout", None),
        getattr(report, "capstderr", None),
    ]))
    if not blob:
        return

    # Prefer the BENCH line when present (training tests); fall back to the
    # inference-style Runtime line. Use the LAST match in each case so any
    # warm-up prints are overridden by the final number.
    bench = None
    for m in _BENCH_RE.finditer(blob):
        bench = m
    runtime = None
    for m in _RUNTIME_CYCLES_RE.finditer(blob):
        runtime = m

    entry: Dict[str, Any] = {
        "nodeid": report.nodeid,
        "outcome": report.outcome,
    }
    if bench is not None:
        entry["train_cycles"] = int(bench.group(1))
        if bench.group(2):
            entry["opt_cycles"] = int(bench.group(2))
        if bench.group(3):
            entry["trainable_bytes"] = int(bench.group(3))
    elif runtime is not None:
        entry["runtime_cycles"] = int(runtime.group(1))
    else:
        return  # no cycle data captured

    _PERF_RESULTS.append(entry)


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:  # noqa: ARG001
    """Print a terminal perf table and write GITHUB_STEP_SUMMARY Markdown."""
    if not _PERF_RESULTS:
        return

    results = sorted(_PERF_RESULTS, key = lambda r: r["nodeid"])

    # --- terminal block ---
    terminalreporter.write_sep("=", "Performance Summary")
    for r in results:
        mark = "PASS" if r["outcome"] == "passed" else "FAIL"
        if "train_cycles" in r:
            extras = f"train={r['train_cycles']:>12,} cyc"
            if "opt_cycles" in r:
                extras += f"  opt={r['opt_cycles']:>10,} cyc"
            if "trainable_bytes" in r:
                extras += f"  trainable_bytes={r['trainable_bytes']:>8,} B"
            terminalreporter.write_line(f"  [{mark}] {r['nodeid']:60s}  {extras}")
        elif "runtime_cycles" in r:
            terminalreporter.write_line(f"  [{mark}] {r['nodeid']:60s}  runtime={r['runtime_cycles']:>12,} cyc")

    # --- GitHub Actions step summary (Markdown) ---
    gh_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not gh_summary:
        return

    has_training = any("train_cycles" in r for r in results)
    has_inference = any("runtime_cycles" in r for r in results)

    lines: List[str] = ["## Performance Summary", ""]

    if has_training:
        lines += [
            "### Training",
            "",
            "| Test | Status | train_cycles | opt_cycles | trainable_bytes |",
            "|---|:---:|---:|---:|---:|",
        ]
        for r in results:
            if "train_cycles" not in r:
                continue
            status = ":white_check_mark:" if r["outcome"] == "passed" else ":x:"
            opt = f"{r['opt_cycles']:,}" if "opt_cycles" in r else "—"
            tb = f"{r['trainable_bytes']:,}" if "trainable_bytes" in r else "—"
            lines.append(f"| `{r['nodeid']}` | {status} | {r['train_cycles']:,} | {opt} | {tb} |")
        lines.append("")

    if has_inference:
        lines += [
            "### Inference",
            "",
            "| Test | Status | Runtime (cycles) |",
            "|---|:---:|---:|",
        ]
        for r in results:
            if "runtime_cycles" not in r:
                continue
            status = ":white_check_mark:" if r["outcome"] == "passed" else ":x:"
            lines.append(f"| `{r['nodeid']}` | {status} | {r['runtime_cycles']:,} |")
        lines.append("")

    try:
        with open(gh_summary, "a") as f:
            f.write("\n".join(lines) + "\n")
    except OSError as e:
        terminalreporter.write_line(f"[perf-summary] Could not write GITHUB_STEP_SUMMARY: {e}")
