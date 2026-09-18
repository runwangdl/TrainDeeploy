# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import os
from abc import ABC
from dataclasses import dataclass
from typing import List, Literal

from Deeploy.DeeployTypes import CodeSnippet, ExecutionBlock, NodeTemplate


@dataclass
class TilingMetaInfo:
    nodeName: str
    nodeOps: int
    numTiles: str
    totalNumTiles: int
    tileIdxPtr: str
    tileIdxVar: str
    kernelLevelTiling: bool


class PrototypeTilingMixIn(ABC):

    @classmethod
    def generateSetupAndTeardownCode(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo,
                                     setupStatements: List[CodeSnippet],
                                     teardownStatements: List[CodeSnippet]) -> ExecutionBlock:

        for transaction in reversed(setupStatements):
            executionBlock.addLeft(transaction.template, transaction.operatorRepresentation)

        for transaction in teardownStatements:
            executionBlock.addRight(transaction.template, transaction.operatorRepresentation)

        return executionBlock

    @classmethod
    def generateLoopCode(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo,
                         openLoopStatements: List[CodeSnippet], ingressDMAStatements: List[CodeSnippet],
                         egressDMAStatements: List[CodeSnippet],
                         closeLoopStatements: List[CodeSnippet]) -> ExecutionBlock:

        for transaction in reversed(openLoopStatements + ingressDMAStatements):
            executionBlock.addLeft(transaction.template, transaction.operatorRepresentation)

        for transaction in egressDMAStatements + closeLoopStatements:
            executionBlock.addRight(transaction.template, transaction.operatorRepresentation)

        return executionBlock

    @classmethod
    def generateAllTilingCode(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo,
                              ingressDMAStatements: List[CodeSnippet], egressDMAStatements: List[CodeSnippet],
                              openLoopStatements: List[CodeSnippet], closeLoopStatements: List[CodeSnippet],
                              setupStatements: List[CodeSnippet],
                              teardownStatements: List[CodeSnippet]) -> ExecutionBlock:

        executionBlock = cls.generateLoopCode(executionBlock, metaInfo, openLoopStatements, ingressDMAStatements,
                                              egressDMAStatements, closeLoopStatements)

        executionBlock = cls.generateSetupAndTeardownCode(executionBlock, metaInfo, setupStatements, teardownStatements)

        return executionBlock


# DEEPLOY_PROFILE_SUMMARY=1: instead of per-tile arrays and one printf per tile, every tile
# loop adds its waits into six global counters (deeploy_prof_acc, defined by the test harness,
# indexed level*3 + {0: ingress wait, 1: kernel, 2: egress wait}, level 0 = L2 loop, 1 = L3
# loop) that the harness prints once. The per-tile mode keeps a few hundred KB of names and
# measurement arrays in L2 and one UART line per tile, which a full training graph cannot
# afford next to tensor promotion; the summary mode costs ~24 B per node.
_PROFILE_SUMMARY = os.environ.get("DEEPLOY_PROFILE_SUMMARY", "0") == "1"


def _summaryAccumulate(measurements: str, tileIdxVar) -> str:
    """C that records one timestamp and, on an *_end timestamp, adds the interval to the
    node level's counter."""
    kind = next(i for i, k in enumerate(("ingress_dma_wait", "kernel", "egress_dma_wait")) if k in measurements)
    level = 1 if f"_L3_{('ingress_dma_wait', 'kernel', 'egress_dma_wait')[kind]}" in measurements else 0
    acc = f"deeploy_prof_acc[{level * 3 + kind}]"
    if not measurements.endswith("_end_measurements"):
        return f"{measurements}[0] = getCycles();"
    start = measurements.replace("_end_measurements", "_start_measurements")
    if kind == 2 and isinstance(tileIdxVar, int):
        # double-buffering teardown: the last tile's egress wait is re-measured after the
        # loop and replaces the in-loop value, so add only the extra wait since then
        return f"{{ uint32_t _t = getCycles(); {acc} += _t - {measurements}[0]; {measurements}[0] = _t; }}"
    return f"{measurements}[0] = getCycles(); {acc} += {measurements}[0] - {start}[0];"


class _MeasureCyclesTemplate(NodeTemplate):

    def generate(self, operatorRepresentation = {}, **kwargs) -> str:
        if _PROFILE_SUMMARY:
            return "\n" + _summaryAccumulate(operatorRepresentation["measurements"],
                                             operatorRepresentation["tileIdxVar"]) + "\n"
        return super().generate(operatorRepresentation, **kwargs)


class ProfilingPrototypeMixIn(ABC):
    _measureCycles = _MeasureCyclesTemplate("""
    ${measurements}[${tileIdxVar}] = getCycles();
    """)

    # RW: 'static' moves the per-node profiling measurement arrays off the CC/master
    # stack (carved from L1) into .bss (-> L2). With a large --l1 arena there is almost
    # no L1 headroom left for the CC stack, so keeping these here would overflow it; the
    # arrays are written/printed master-side and sequentially per node, so a single
    # static instance is correct. Lets --profileTiling run at the full L1 arena size.
    _measurementArrayDeclaration = NodeTemplate("""
    static uint32_t ${measurements}[${totalNumTiles}];
    """)

    _stringDeclaration = NodeTemplate("""
    const static char ${name}[] = "${string}";
    """)

    _printLoopSetup = NodeTemplate("""
    StopTimer();
    printf("===== Profiling ${nodeName} =====\\n");
    for (int ${profileIdxVar} = ((*${tileIdxPtr} > 0) ? ${numTiles}[(*${tileIdxPtr} - 1)] : 0);
        ${profileIdxVar} < ${numTiles}[*${tileIdxPtr}];
        ${profileIdxVar}++){
    """)

    _measurementDeclaration = NodeTemplate("""
    uint32_t ${measurement} = ${measurementsEnd}[${profileIdxVar}] - ${measurementsStart}[${profileIdxVar}];
    """)

    _printCycleDifference = NodeTemplate("""
    printf("%s%u] %s%6u%s", ${prefixStr}, ${profileIdxVar}, "${flavorStr}", \
    ${measurement}, ${suffixStr});
    """)

    _printCycleContribution = NodeTemplate("""
    uint32_t total = ${measurementInput} + ${measurementKernel} + ${measurementOutput};
    uint32_t dma = ${measurementInput} + ${measurementOutput};
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : ${measurementKernel} * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\\n", ${prefixStr}, ${profileIdxVar}, total, kernel_percentage, overhead_percentage    , ${measurementKernel}, dma);
    """)

    _printLoopTeardown = NodeTemplate("""
    }
    StartTimer();
    """)

    _measureConditionSetup = NodeTemplate("""
    if(${cond}){
    """)

    _measureConditionEnd = NodeTemplate("""
    }
    """)

    @classmethod
    def measurementArrayDeclaration(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo,
                                    bufferingStr: Literal["SB", "DB"]) -> ExecutionBlock:

        nodeName = metaInfo.nodeName
        numTiles = metaInfo.numTiles
        totalNumTiles = metaInfo.totalNumTiles
        nodeOps = metaInfo.nodeOps

        measurementsList = [
            "ingress_dma_wait_start", "ingress_dma_wait_end", "egress_dma_wait_start", "egress_dma_wait_end"
        ]

        if metaInfo.kernelLevelTiling:
            measurementsList = ["kernel_start", "kernel_end"] + measurementsList

        for measurements in measurementsList:
            executionBlock.addLeft(cls._measurementArrayDeclaration, {
                "measurements": f"{nodeName}_{measurements}_measurements",
                "totalNumTiles": 1 if _PROFILE_SUMMARY else totalNumTiles
            })

        if _PROFILE_SUMMARY:
            executionBlock.addLeft(NodeTemplate("extern uint32_t deeploy_prof_acc[6];\n"), {})
            return executionBlock

        executionBlock.addLeft(cls._stringDeclaration, {
            "name": f"{nodeName}_prefix",
            "string": f"[{nodeName}][{bufferingStr}][{nodeOps} ops][Tile ",
        })

        executionBlock.addLeft(cls._stringDeclaration, {
            "name": f"{nodeName}_suffix",
            "string": " cycles \\n",
        })

        return executionBlock

    @classmethod
    def injectPrintCycleDiff(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo) -> ExecutionBlock:

        if _PROFILE_SUMMARY:  # printed once by the harness
            return executionBlock

        numTiles = metaInfo.numTiles
        nodeName = metaInfo.nodeName
        tileIdxPtr = metaInfo.tileIdxPtr
        profileIdxVar = "PROFILING_I"

        executionBlock.addRight(cls._printLoopSetup, {
            "numTiles": numTiles,
            "nodeName": nodeName,
            "profileIdxVar": profileIdxVar,
            "tileIdxPtr": tileIdxPtr,
        })

        executionBlock.addRight(
            cls._measurementDeclaration, {
                "measurement": f"{nodeName}_ingress_dma_wait_measurement",
                "measurementsStart": f"{nodeName}_ingress_dma_wait_start_measurements",
                "measurementsEnd": f"{nodeName}_ingress_dma_wait_end_measurements",
                "profileIdxVar": profileIdxVar,
            })

        if metaInfo.kernelLevelTiling:
            executionBlock.addRight(
                cls._measurementDeclaration, {
                    "measurement": f"{nodeName}_kernel_measurement",
                    "measurementsStart": f"{nodeName}_kernel_start_measurements",
                    "measurementsEnd": f"{nodeName}_kernel_end_measurements",
                    "profileIdxVar": profileIdxVar,
                })

        executionBlock.addRight(
            cls._measurementDeclaration, {
                "measurement": f"{nodeName}_egress_dma_wait_measurement",
                "measurementsStart": f"{nodeName}_egress_dma_wait_start_measurements",
                "measurementsEnd": f"{nodeName}_egress_dma_wait_end_measurements",
                "profileIdxVar": profileIdxVar,
            })

        executionBlock.addRight(
            cls._printCycleDifference, {
                "prefixStr": f"{nodeName}_prefix",
                "suffixStr": f"{nodeName}_suffix",
                "flavorStr": "Pre-Kernel :",
                "measurement": f"{nodeName}_ingress_dma_wait_measurement",
                "profileIdxVar": profileIdxVar,
            })

        if metaInfo.kernelLevelTiling:
            executionBlock.addRight(
                cls._printCycleDifference, {
                    "prefixStr": f"{nodeName}_prefix",
                    "suffixStr": f"{nodeName}_suffix",
                    "flavorStr": "Kernel     :",
                    "measurement": f"{nodeName}_kernel_measurement",
                    "profileIdxVar": profileIdxVar,
                })

        executionBlock.addRight(
            cls._printCycleDifference, {
                "prefixStr": f"{nodeName}_prefix",
                "suffixStr": f"{nodeName}_suffix",
                "flavorStr": "Post-Kernel:",
                "measurement": f"{nodeName}_egress_dma_wait_measurement",
                "profileIdxVar": profileIdxVar,
            })

        # Total Time: Input + Kernel + Output
        # Overhead: (Input + Output) / Total
        if metaInfo.kernelLevelTiling:
            executionBlock.addRight(
                cls._printCycleContribution, {
                    "prefixStr": f"{nodeName}_prefix",
                    "measurementInput": f"{nodeName}_ingress_dma_wait_measurement",
                    "measurementKernel": f"{nodeName}_kernel_measurement",
                    "measurementOutput": f"{nodeName}_egress_dma_wait_measurement",
                    "profileIdxVar": profileIdxVar,
                })

        executionBlock.addRight(cls._printLoopTeardown, {})

        return executionBlock

    @classmethod
    def kernelProfilingWrap(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo) -> ExecutionBlock:
        nodeName = metaInfo.nodeName
        tileIdxVar = metaInfo.tileIdxVar

        if metaInfo.kernelLevelTiling:
            executionBlock.addLeft(cls._measureCycles, {
                "measurements": f"{nodeName}_kernel_start_measurements",
                "tileIdxVar": tileIdxVar
            })
            executionBlock.addRight(cls._measureCycles, {
                "measurements": f"{nodeName}_kernel_end_measurements",
                "tileIdxVar": tileIdxVar
            })

        return executionBlock
