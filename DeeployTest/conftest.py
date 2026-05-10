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
# Training cycle summary: at session end, scan $GITHUB_STEP_SUMMARY for any
# training cycle section emitted by run_and_assert_test, join SB and DB rows
# by (test, l1), and append a comparison table with speedup.
# ---------------------------------------------------------------------------
def _parse_training_section(section_body: str):
    """Parse rows of `| test | l1 | mode | train_cycles | opt_cycles | weight_sram |`.

    Returns list of dicts with keys: test, l1, mode, train, opt, sram.
    """
    rows = []
    for line in section_body.splitlines():
        if not line.startswith("| "):
            continue
        if "train_cycles" in line or "------" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 6:
            continue
        try:
            train = int(cells[3].replace(",", ""))
            opt = int(cells[4].replace(",", ""))
            sram = int(cells[5].replace(",", ""))
        except ValueError:
            continue
        rows.append({"test": cells[0], "l1": cells[1], "mode": cells[2], "train": train, "opt": opt, "sram": sram})
    return rows


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path or not os.path.exists(summary_path):
        return
    try:
        with open(summary_path, "r") as f:
            existing = f.read()
    except Exception:
        return

    # Find every "## Siracusa L? training cycles" section and append a join.
    import re as _re
    for heading in _re.findall(r"^## (Siracusa L[23] training cycles)$", existing, flags = _re.MULTILINE):
        start = existing.find(f"## {heading}")
        rest = existing[start + len(f"## {heading}"):]
        next_section = rest.find("\n## ")
        body = rest if next_section == -1 else rest[:next_section]
        rows = _parse_training_section(body)
        if not rows:
            continue
        # Join SB and DB rows by (test, l1).
        by_key: dict = {}
        for r in rows:
            by_key.setdefault((r["test"], r["l1"]), {})[r["mode"]] = r
        try:
            with open(summary_path, "a") as f:
                f.write(f"\n### {heading} — SB vs DB speedup\n\n")
                f.write("| Test | L1 (B) | SB train | DB train | train Δ | SB opt | DB opt | opt Δ |\n")
                f.write("|------|--------|----------|----------|---------|--------|--------|-------|\n")
                for (test, l1), modes in sorted(by_key.items()):
                    sb = modes.get("SB")
                    db = modes.get("DB")
                    sb_t = f"{sb['train']:,}" if sb else "—"
                    db_t = f"{db['train']:,}" if db else "—"
                    sb_o = f"{sb['opt']:,}" if sb else "—"
                    db_o = f"{db['opt']:,}" if db else "—"
                    if sb and db and sb['train'] > 0:
                        delta_t = f"{(sb['train'] - db['train']) / sb['train'] * 100:+.1f}%"
                    else:
                        delta_t = "—"
                    if sb and db and sb['opt'] > 0:
                        delta_o = f"{(sb['opt'] - db['opt']) / sb['opt'] * 100:+.1f}%"
                    else:
                        delta_o = "—"
                    f.write(f"| {test} | {l1} | {sb_t} | {db_t} | {delta_t} | {sb_o} | {db_o} | {delta_o} |\n")
        except Exception:
            pass
