# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Dict, List, Literal, Optional

from .core import DeeployTestConfig, build_binary, configure_cmake, get_test_paths, run_complete_test, run_simulation

__all__ = [
    'get_worker_id',
    'create_test_config',
    'run_and_assert_test',
    'build_binary',
    'configure_cmake',
    'run_simulation',
]


def get_worker_id() -> str:
    """
    Get the pytest-xdist worker ID for parallel test execution.

    Returns:
        Worker ID string (e.g., 'gw0', 'gw1', 'master' for non-parallel)
    """
    return os.environ.get("PYTEST_XDIST_WORKER", "master")


def create_test_config(
    test_name: str,
    platform: str,
    simulator: Literal['gvsoc', 'banshee', 'qemu', 'vsim', 'vsim.gui', 'host', 'board', 'none'],
    deeploy_test_dir: str,
    toolchain: str,
    toolchain_dir: Optional[str],
    cmake_args: List[str],
    tiling: bool = False,
    cores: Optional[int] = None,
    l1: Optional[int] = None,
    l2: int = 1024000,
    default_mem_level: str = "L2",
    double_buffer: bool = False,
    mem_alloc_strategy: str = "MiniMalloc",
    search_strategy: str = "random-max",
    profile_tiling: bool = False,
    profile_microbenchmark: bool = False,
    plot_mem_alloc: bool = False,
    randomized_mem_scheduler: bool = False,
    profile_untiled: bool = False,
    training: bool = False,
    training_num_data_inputs: Optional[int] = None,
    training_tolerance: Optional[float] = None,
    promote_to_l2: bool = False,
    promote_to_l2_strategy: str = "cycle-aware",
    promote_to_l2_headroom: int = 131072,
    gen_args: Optional[List[str]] = None,
) -> DeeployTestConfig:

    test_dir = f"Tests/{test_name}"

    gen_dir, test_dir_abs, test_name_clean = get_test_paths(test_dir, platform, base_dir = deeploy_test_dir)

    worker_id = get_worker_id()

    if worker_id == "master":
        build_dir = str(Path(deeploy_test_dir) / f"TEST_{platform.upper()}" / "build_master")
    else:
        build_dir = str(Path(deeploy_test_dir) / f"TEST_{platform.upper()}" / f"build_{worker_id}")

    cmake_args_list = list(cmake_args) if cmake_args else []
    if cores is not None:
        cmake_args_list.append(f"NUM_CORES={cores}")

    gen_args_list = list(gen_args) if gen_args else []

    if cores is not None and platform in ["Siracusa", "Siracusa_w_neureka"]:
        gen_args_list.append(f"--cores={cores}")

    if tiling:
        if l1 is not None:
            gen_args_list.append(f"--l1={l1}")
        if l2 != 1024000:
            gen_args_list.append(f"--l2={l2}")
        if default_mem_level != "L2":
            gen_args_list.append(f"--defaultMemLevel={default_mem_level}")
        if double_buffer:
            gen_args_list.append("--doublebuffer")
        if mem_alloc_strategy != "MiniMalloc":
            gen_args_list.append(f"--memAllocStrategy={mem_alloc_strategy}")
        if search_strategy != "random-max":
            gen_args_list.append(f"--searchStrategy={search_strategy}")
        if profile_tiling:
            gen_args_list.append("--profileTiling")
        if profile_microbenchmark:
            gen_args_list.append("--profileMicrobenchmark")
        if plot_mem_alloc:
            gen_args_list.append("--plotMemAlloc")
        if randomized_mem_scheduler:
            gen_args_list.append("--randomizedMemoryScheduler")
        if promote_to_l2:
            assert default_mem_level == "L3", "promote_to_l2 requires default_mem_level='L3'"
            gen_args_list.append("--promoteToL2")
            gen_args_list.append(f"--promoteToL2Strategy={promote_to_l2_strategy}")
            gen_args_list.append("--promoteToL2IncludeActivations")
            gen_args_list.append("--promoteToL2MaxBufferBytes=0")
            gen_args_list.append(f"--promoteToL2Headroom={promote_to_l2_headroom}")

    if profile_untiled and not tiling and platform == "Siracusa":
        gen_args_list.append("--profileUntiled")

    if training and training_tolerance is not None:
        gen_args_list.append(f"--tolerance={training_tolerance}")

    config = DeeployTestConfig(
        test_name = test_name_clean,
        test_dir = test_dir_abs,
        platform = platform,
        simulator = simulator,
        tiling = tiling,
        gen_dir = gen_dir,
        build_dir = build_dir,
        toolchain = toolchain,
        toolchain_install_dir = toolchain_dir,
        cmake_args = cmake_args_list,
        gen_args = gen_args_list,
        training = training,
        training_num_data_inputs = training_num_data_inputs,
    )

    return config


# Track which Markdown section headers we've already written into
# $GITHUB_STEP_SUMMARY so each is emitted exactly once per pytest session
# even though many tests share the same metric_section.
_METRIC_SECTIONS_WRITTEN: set = set()


def _emit_metric_section_header(summary_path: str, section: str, columns: list) -> None:
    if section in _METRIC_SECTIONS_WRITTEN:
        return
    _METRIC_SECTIONS_WRITTEN.add(section)
    try:
        with open(summary_path, "a") as f:
            f.write(f"\n## {section}\n\n")
            f.write("| " + " | ".join(columns) + " |\n")
            f.write("|" + "|".join(["------"] * len(columns)) + "|\n")
    except Exception:
        pass


def run_and_assert_test(test_name: str,
                        config: DeeployTestConfig,
                        skipgen: bool,
                        skipsim: bool,
                        report_metric: Optional[Dict[str, str]] = None,
                        metric_section: str = "Tensor-promotion strategy benchmark") -> None:
    """
    Shared helper function to run a test and assert its results.

    If ``report_metric`` is given (a dict of label -> value) and a cycle count
    is parseable from stdout, append a row to ``$GITHUB_STEP_SUMMARY`` (under
    the ``metric_section`` heading) so the metric shows up in the GitHub
    Actions run summary, and print a tagged line that survives pytest's
    stdout capture.

    ``metric_section`` lets unrelated test categories (e.g. promotion
    benchmark vs. training cycle reference) write to separate Markdown
    tables in the same workflow summary.

    Raises:
        AssertionError: If test fails or has errors
    """
    result = run_complete_test(config, skipgen = skipgen, skipsim = skipsim)

    cycles = getattr(result, "runtime_cycles", None)
    if cycles is None and getattr(result, "stdout", None):
        # Training tests emit "BENCH train_cycles=N opt_cycles=M weight_sram=K"
        # instead of "Runtime: N cycles"; fall back to that format so the
        # training cycle reference table works the same as the inference one.
        import re as _re
        m = _re.search(r'BENCH\s+train_cycles=(\d+)', result.stdout)
        if m:
            cycles = int(m.group(1))

    if report_metric is not None:
        labels = " ".join(f"{k}={v}" for k, v in report_metric.items())
        cycles_str = f"{cycles:,}" if cycles is not None else "n/a"
        # Always print a clearly-tagged line; pytest captures stdout but shows
        # it on failure, and `-rA` (used in CI) shows captured output for
        # passing tests too. Embed the section so log greppers can group.
        print(f"\n[METRIC] section={metric_section!r} test={test_name} {labels} cycles={cycles_str}", flush = True)
        # Append a Markdown table row to GITHUB_STEP_SUMMARY when running in
        # GitHub Actions; the file is auto-created and rendered as Markdown
        # in the workflow summary panel. The first row in each section also
        # writes a heading so the table renders correctly.
        summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary_path and cycles is not None:
            columns = ["Test"] + list(report_metric.keys()) + ["Cycles"]
            _emit_metric_section_header(summary_path, metric_section, columns)
            row_cells = [test_name] + [str(v) for v in report_metric.values()] + [f"{cycles:,}"]
            try:
                with open(summary_path, "a") as f:
                    f.write("| " + " | ".join(row_cells) + " |\n")
            except Exception:
                pass

    assert result.success, (f"Test {test_name} failed with {result.error_count} errors out of {result.total_count}\n"
                            f"Output:\n{result.stdout}")

    if result.error_count >= 0:
        assert result.error_count == 0, (f"Found {result.error_count} errors out of {result.total_count} tests")
