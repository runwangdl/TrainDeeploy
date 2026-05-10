# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import coloredlogs
import pytest

from Deeploy.Logging import DEFAULT_FMT
from Deeploy.Logging import DEFAULT_LOGGER as log


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
# Promotion-strategy benchmark: Markdown summary written to
# $GITHUB_STEP_SUMMARY so the GitHub Actions UI shows a comparison table
# directly in the workflow run summary panel.
# ---------------------------------------------------------------------------

PROMOTION_METRIC_HEADER_DONE = False


def _promotion_summary_path():
    return os.environ.get("GITHUB_STEP_SUMMARY")


@pytest.fixture(autouse = True)
def _promotion_summary_header():
    """Emit a Markdown table header once per session if running under GH Actions
    AND the session contains promotion tests. The actual rows are appended by
    pytestRunner.run_and_assert_test as each parametrised case finishes."""
    global PROMOTION_METRIC_HEADER_DONE
    path = _promotion_summary_path()
    if path and not PROMOTION_METRIC_HEADER_DONE:
        try:
            with open(path, "a") as f:
                f.write("\n## Tensor-promotion strategy benchmark\n\n")
                f.write("| Test | Strategy | Activations | L1 (B) | Cycles |\n")
                f.write("|------|----------|-------------|--------|--------|\n")
            PROMOTION_METRIC_HEADER_DONE = True
        except Exception:
            pass


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """At the end of the session, recompute the savings table from the
    Markdown rows already written by individual tests and append a second
    table that shows ``cycles`` and ``Δ vs off`` so reviewers see deltas
    without doing arithmetic."""
    path = _promotion_summary_path()
    if not path or not os.path.exists(path):
        return
    try:
        existing = open(path).read()
    except Exception:
        return
    if "## Tensor-promotion strategy benchmark" not in existing:
        return

    # Parse the Markdown rows we appended (after the header).
    rows = []
    for line in existing.splitlines():
        if not line.startswith("| ") or "Cycles" in line or "---" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 5:
            continue
        try:
            cycles = int(cells[-1].replace(",", ""))
        except ValueError:
            continue
        rows.append((cells[0], cells[1], cells[2], cells[3], cycles))

    # Find the "off" baseline per (test, l1) pair
    baselines = {(t, l1): cyc for t, strat, _, l1, cyc in rows if strat == "off"}
    if not rows or not baselines:
        return

    try:
        with open(path, "a") as f:
            f.write("\n### Cycle savings vs `off` baseline\n\n")
            f.write("| Test | Strategy | Activations | Cycles | Δ vs baseline |\n")
            f.write("|------|----------|-------------|--------|---------------|\n")
            for t, strat, acts, l1, cyc in rows:
                base = baselines.get((t, l1))
                if base is None:
                    continue
                if strat == "off":
                    delta = "—"
                else:
                    pct = (cyc - base) / base * 100
                    delta = f"{pct:+.2f}%"
                f.write(f"| {t} | `{strat}` | {acts} | {cyc:,} | {delta} |\n")
            f.write("\n")
    except Exception:
        pass
