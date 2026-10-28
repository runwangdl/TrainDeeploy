# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import os
import sys
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest
from testUtils.codeGenerate import generateTestNetwork
from testUtils.graphDebug import generateDebugConfig
from testUtils.platformMapping import mapDeployer, mapPlatform, setupMemoryPlatform
from testUtils.testRunner import TestGeneratorArgumentParser
from testUtils.tilingUtils import DBOnlyL3Tiler, DBTiler, SBTiler
from testUtils.typeMapping import inferTypeAndOffset, parseDataType  # -- QW: parseDataType for --input-type-map

from Deeploy.AbstractDataTypes import PointerClass  # -- QW: for --input-type-map overrides
from Deeploy.DeeployTypes import CodeGenVerbosity, NetworkDeployer, ONNXLayer
from Deeploy.EngineExtension.NetworkDeployers.EngineColoringDeployer import EngineColoringDeployerWrapper
from Deeploy.Logging import DEFAULT_LOGGER as log
from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryHierarchy, MemoryLevel
from Deeploy.MemoryLevelExtension.NetworkDeployers.MemoryLevelDeployer import MemoryDeployerWrapper
from Deeploy.MemoryLevelExtension.OptimizationPasses.MemoryLevelAnnotationPasses import AnnotateDefaultMemoryLevel, \
    AnnotateIOMemoryLevel, PromoteTensorsToL2
from Deeploy.Targets.Neureka.OptimizationPasses.MemoryLevelAnnotationPasses import AnnotateNeurekaWeightMemoryLevel
from Deeploy.Targets.PULPOpen.Platform import PULPClusterEngine
from Deeploy.TilingExtension.TilerExtension import TilerDeployerWrapper


def _resolveInputTypes(args, npzNames, testInputs, signProp):  # -- QW
    """Per-input (type, offset), honouring --input-type-map/--input-offset-map. -- QW

    The TILED generator was missing the override that generateNetwork.py has, and it is
    needed here: `inferTypeAndOffset` picks the *minimal* float type that fits the data
    (`minimalFloatType` walks bfloat16 -> float16 -> float32), so a small-magnitude fp32
    tensor -- e.g. SpeechNet block-1's conv bias, |b| < 1e-5 -- is inferred as bfloat16 and
    then has no binding at all, because the fp32 kernels are float32-only.

    The ONNX file does declare FLOAT, so testMVPTraining.py's approach (trust the declared
    elem_type) would fix it -- but 32 Integer kernel tests declare FLOAT while carrying
    int8/int64 npz data, so that trust cannot be extended to this path. An opt-in override
    keyed by npz entry name is the safe mechanism.
    """
    manualTypes, manualOffsets = {}, {}
    for kv in getattr(args, "input_type_map", []) or []:
        name, tstr = kv.split("=", 1)
        manualTypes[name.strip()] = parseDataType(tstr.strip())
    for kv in getattr(args, "input_offset_map", []) or []:
        name, ostr = kv.split("=", 1)
        manualOffsets[name.strip()] = int(ostr.strip())

    assert set(manualTypes) == set(manualOffsets), \
        f"Override inputs need both a type and an offset. Mismatch: {set(manualTypes) ^ set(manualOffsets)}"
    assert set(manualTypes) <= set(npzNames), \
        f"Unknown input names in overrides: {set(manualTypes) - set(npzNames)} (valid: {set(npzNames)})"

    inputTypes, inputOffsets = {}, {}
    for index, values in enumerate(testInputs):
        name = npzNames[index] if index < len(npzNames) else None
        if name in manualTypes:
            inputTypes[f"input_{index}"] = PointerClass(manualTypes[name])
            inputOffsets[f"input_{index}"] = manualOffsets[name]
        else:
            inputTypes[f"input_{index}"], inputOffsets[f"input_{index}"] = \
                inferTypeAndOffset(values, signProp)
    return inputTypes, inputOffsets


# Mock of the Global Scheduler's inteface
# Returns a list of list of nodes instead of simply a list
# Inner list represent the patter over which we tile
def _mockScheduler(graph: gs.Graph) -> List[List[gs.Node]]:

    schedule = [[node] for node in graph.nodes]

    return schedule


def _filterSchedule(schedule: List[List[gs.Node]], layerBinding: 'OrderedDict[str, ONNXLayer]') -> List[List[gs.Node]]:

    filteredSchedule = []

    for pattern in schedule:

        filteredSchedulePattern = []
        for node in pattern:
            if node.name in layerBinding.keys():
                filteredSchedulePattern.append(node)
        filteredSchedule.append(filteredSchedulePattern)

    return filteredSchedule


def setupDeployer(graph: gs.Graph, memoryHierarchy: MemoryHierarchy, defaultTargetMemoryLevel: MemoryLevel,
                  defaultIoMemoryLevel: MemoryLevel, verbose: CodeGenVerbosity,
                  args: argparse.Namespace) -> Tuple[NetworkDeployer, bool]:

    inputTypes = {}
    inputOffsets = {}

    _DEEPLOYSTATEDIR = os.path.join(args.dumpdir, "deeployStates")

    inputs = np.load(f'{args.dir}/inputs.npz')
    tensors = graph.tensors()

    # Load as int64 and infer types later
    test_inputs = [inputs[x].reshape(-1).astype(np.float64) for x in inputs.files]

    platform, signProp = mapPlatform(args.platform)

    if args.enable_3x3:
        platform.engines[0].enable3x3 = True
    if args.enableStrides:
        platform.engines[0].enableStrides = True
    if getattr(args, "enable_1xk", False):  # -- QW (exp16a): 1xK -> K pointwise NE16 dispatches
        platform.engines[0].enable1xK = True  # -- QW

    clusters = [engine for engine in platform.engines if isinstance(engine, PULPClusterEngine)]
    for cluster in clusters:
        cluster.n_cores = args.cores

    inputTypes, inputOffsets = _resolveInputTypes(args, inputs.files, test_inputs, signProp)  # -- QW

    deployer = mapDeployer(platform,
                           graph,
                           inputTypes,
                           deeployStateDir = _DEEPLOYSTATEDIR,
                           inputOffsets = inputOffsets,
                           scheduler = _mockScheduler,
                           conv_channels_first = getattr(args, "convChannelsFirst", False))

    # Make the deployer engine-color-aware
    if args.platform in ("Siracusa_w_neureka", "GAP9_w_NE16"):  # -- QW: composite platforms
        deployer = EngineColoringDeployerWrapper(deployer)

    # Make platform memory-aware after mapDeployer because it requires the platform to be an instance of an unwrapped platform
    deployer.Platform = setupMemoryPlatform(deployer.Platform, memoryHierarchy, defaultTargetMemoryLevel)

    memoryLevelAnnotationPasses = [
        AnnotateIOMemoryLevel(defaultIoMemoryLevel.name),
        AnnotateDefaultMemoryLevel(memoryHierarchy)
    ]

    if args.neureka_wmem:
        weightMemoryLevel = memoryHierarchy.memoryLevels["WeightMemory_SRAM"]
        memoryLevelAnnotationPasses.append(
            AnnotateNeurekaWeightMemoryLevel(neurekaEngineName = deployer.Platform.engines[0].name,
                                             weightMemoryLevel = weightMemoryLevel))

    if args.promoteToL2:
        assert args.defaultMemLevel == "L3", "--promoteToL2 only makes sense when --defaultMemLevel L3"
        memoryLevelAnnotationPasses.append(
            PromoteTensorsToL2(
                l2Size = memoryHierarchy.memoryLevels["L2"].size,
                headroom = args.promoteToL2Headroom,
                strategy = args.promoteToL2Strategy,
                includeActivations = args.promoteToL2IncludeActivations,
                maxBufferBytes = args.promoteToL2MaxBufferBytes,
            ))

    # Make the deployer memory-level aware
    deployer = MemoryDeployerWrapper(deployer, memoryLevelAnnotationPasses)

    # Make the deployer tiler aware
    # VJUNG: Create unique ID for the IO files of minimalloc and prevent conflict in case of parallel execution
    unique_params = f"{args.dumpdir}_L1{args.l1}_L2{args.l2}_{args.defaultMemLevel}_DB{args.doublebuffer}"
    testIdentifier = hashlib.md5(unique_params.encode()).hexdigest()[:16]

    if args.doublebuffer:
        assert args.defaultMemLevel in ["L3", "L2"]
        if args.defaultMemLevel == "L3":
            deployer = TilerDeployerWrapper(deployer, DBOnlyL3Tiler, testName = testIdentifier, workDir = args.dumpdir)
        else:
            deployer = TilerDeployerWrapper(deployer, DBTiler, testName = testIdentifier, workDir = args.dumpdir)
    else:
        deployer = TilerDeployerWrapper(deployer, SBTiler, testName = testIdentifier, workDir = args.dumpdir)

    deployer.tiler.visualizeMemoryAlloc = args.plotMemAlloc
    deployer.tiler.memoryAllocStrategy = args.memAllocStrategy
    deployer.tiler.searchStrategy = args.searchStrategy

    return deployer, signProp


if __name__ == '__main__':

    parser = TestGeneratorArgumentParser(
        description = "Deeploy Code Generation Utility with Memory Level Annotation and Tiling Extension.")

    parser.add_argument('--debug',
                        dest = 'debug',
                        action = 'store_true',
                        default = False,
                        help = 'Enable debugging mode\n')
    parser.add_argument('--defaultMemLevel',
                        metavar = 'defaultMemLevel',
                        dest = 'defaultMemLevel',
                        type = str,
                        default = "L2",
                        help = 'Set default memory level\n')
    parser.add_argument('--neureka-wmem',
                        dest = "neureka_wmem",
                        action = "store_true",
                        default = False,
                        help = 'Adds weight memory and neureka engine color\n')
    parser.add_argument('--enable-3x3',
                        dest = "enable_3x3",
                        action = "store_true",
                        default = False,
                        help = 'Adds EXPERIMENTAL support for 3x3 convolutions on N-EUREKA\n')
    parser.add_argument('--enable-1xk',
                        dest = "enable_1xk",
                        action = "store_true",
                        default = False,
                        help = 'Let NE16 claim 1xK / Kx1 dense convs and run them as K pointwise '
                        'dispatches accumulating via streamin (exp16a / STEP 2b)\n')
    parser.add_argument('--enableStrides',
                        dest = "enableStrides",
                        action = "store_true",
                        default = False,
                        help = 'Adds EXPERIMENTAL support for strided convolutions on N-EUREKA\n')
    parser.add_argument('--doublebuffer', action = 'store_true')
    parser.add_argument('--convChannelsFirst',
                        action = 'store_true',
                        default = False,
                        help = 'Keep forward convs channels-first (NCHW) and bind to the *_CHW kernels (GAP9).')
    parser.add_argument('--l1',
                        metavar = 'l1',
                        dest = 'l1',
                        type = int,
                        default = 64000,
                        help = 'Set L1 size in bytes. \n')
    parser.add_argument('--l2',
                        metavar = 'l2',
                        dest = 'l2',
                        type = int,
                        default = 1024000,
                        help = 'Set L2 size in bytes.\n')
    parser.add_argument('--shouldFail', action = 'store_true')
    parser.add_argument('--memAllocStrategy',
                        metavar = 'memAllocStrategy',
                        dest = 'memAllocStrategy',
                        type = str,
                        default = "MiniMalloc",
                        help = """Choose the memory allocation strategy, possible values are:
                            - TetrisRandom: Randomly sample an placement schedule (order) for the Tetris Memory Allocation.
                            - TetrisCo-Opt: Co-optimize the placement schedule with the tiling solver (works best with random-max solver strategy).
                            - MiniMalloc: Use SotA static memory allocator from https://dl.acm.org/doi/10.1145/3623278.3624752
                        """)
    parser.add_argument('--searchStrategy',
                        metavar = 'searchStrategy',
                        dest = 'searchStrategy',
                        type = str,
                        default = "random-max",
                        help = """Choose the search strategy for the CP solver:
                            - random-max: Initalize the permutation matrix variables randomly and initalize all other variables at their maximal value. This is recommended and lead to better solutions.
                            - max: Initalize all variables at their maximal value.
                            - min: Initalize all variables at their minimal value.
                        """)
    parser.add_argument('--profileTiling', action = "store_true", help = 'Enable tiling profiling')
    parser.add_argument('--profileMicrobenchmark',
                        action = "store_true",
                        help = 'Wrap each layer with PULP perf-counter microbenchmark instrumentation')
    parser.add_argument('--plotMemAlloc',
                        action = 'store_true',
                        help = 'Turn on plotting of the memory allocation and save it in the deeployState folder\n')
    parser.add_argument('--input-type-map',
                        nargs = '*',
                        default = [],
                        type = str,
                        help = '(Optional) map npz input names to data types, overriding value '
                        'inference. Pair with --input-offset-map. '
                        'Example: --input-type-map blocks_1_0_bias=float32_t')  # -- QW
    parser.add_argument('--input-offset-map',
                        nargs = '*',
                        default = [],
                        type = str,
                        help = '(Optional) map npz input names to offsets (0 for float types). '
                        'Example: --input-offset-map blocks_1_0_bias=0')  # -- QW
    parser.add_argument(
        "--cores",
        type = int,
        default = 1,
        help =
        "Number of cores on which the network is run. Currently, required for im2col buffer sizing on Siracusa. Default: 1."
    )
    parser.add_argument('--promoteToL2',
                        action = 'store_true',
                        help = 'Promote selected L3 tensors to L2 (requires --defaultMemLevel L3).')
    parser.add_argument('--promoteToL2Strategy',
                        type = str,
                        default = 'cycle-aware',
                        choices = [
                            'traffic-per-peak', 'cycle-aware', 'greedy-score', 'knapsack-ratio', 'smallest', 'largest',
                            'random'
                        ],
                        help = 'Selection strategy for PromoteTensorsToL2. Default: cycle-aware.')
    parser.add_argument('--promoteToL2IncludeActivations',
                        action = 'store_true',
                        default = True,
                        help = 'Include activations and graph I/O as promotion candidates (default: True).')
    parser.add_argument('--promoteToL2MaxBufferBytes',
                        type = int,
                        default = 0,
                        help = 'Skip candidates larger than this many bytes; 0 = no cap. Default: 0.')
    parser.add_argument('--promoteToL2Headroom',
                        type = int,
                        default = 131072,
                        help = 'Bytes reserved in L2 for tile staging (not available for promotion). Default: 131072.')

    parser.set_defaults(shouldFail = False)
    args = parser.parse_args()

    log.debug("Arguments: %s", args)

    verbosityCfg = CodeGenVerbosity(None)

    if args.profileTiling:
        verbosityCfg.tilingProfiling = True

    if args.profileMicrobenchmark:
        verbosityCfg.microbenchmarkProfiling = True

    onnx_graph = onnx.load_model(f'{args.dir}/network.onnx')
    graph = gs.import_onnx(onnx_graph)

    inputTypes = {}
    inputOffsets = {}

    inputs = np.load(f'{args.dir}/inputs.npz')
    outputs = np.load(f'{args.dir}/outputs.npz')
    if os.path.isfile(f'{args.dir}/activations.npz'):
        activations = np.load(f'{args.dir}/activations.npz')
    else:
        activations = None

    tensors = graph.tensors()

    if args.debug:
        test_inputs, test_outputs, graph = generateDebugConfig(inputs, outputs, activations, graph)
    else:
        # Load as int64 and infer types later
        test_inputs = [inputs[x].reshape(-1).astype(np.float64) for x in inputs.files]
        test_outputs = [outputs[x].reshape(-1).astype(np.float64) for x in outputs.files]

        # WIESEP: Hack to get CI running because only one specific array is used
        if "WaveFormer" in args.dir:
            test_inputs = [test_inputs[0]]
            test_outputs = [test_outputs[-2]]

    # Instantiate Classes Requried for Memory Level Annotation Extension
    L3 = MemoryLevel(name = "L3", neighbourNames = ["L2"], size = 64000000)
    L2 = MemoryLevel(name = "L2", neighbourNames = ["L3", "L1"], size = args.l2)
    L1 = MemoryLevel(name = "L1", neighbourNames = ["L2"], size = args.l1)
    memoryLevels = [L3, L2, L1]

    if args.neureka_wmem:
        memoryLevels.append(MemoryLevel(name = "WeightMemory_SRAM", neighbourNames = [], size = 4 * 1024 * 1024))

    memoryHierarchy = MemoryHierarchy(memoryLevels)
    memoryHierarchy.setDefaultMemoryLevel(args.defaultMemLevel)

    deployer, signProp = setupDeployer(graph,
                                       memoryHierarchy,
                                       defaultTargetMemoryLevel = L1,
                                       defaultIoMemoryLevel = memoryHierarchy.memoryLevels[args.defaultMemLevel],
                                       verbose = verbosityCfg,
                                       args = args)

    platform = deployer.Platform

    log.debug(f"Platform: {platform} (sign: {signProp})")

    log.debug("Platform Engines:")
    for engine in platform.engines:
        log.debug(f" - {engine.name}: {engine}")

    log.debug(f"Deployer: {deployer}")

    inputTypes, inputOffsets = _resolveInputTypes(args, inputs.files, test_inputs, signProp)  # -- QW

    schedule = _filterSchedule(_mockScheduler(graph), deployer.layerBinding)

    if args.shouldFail:
        with pytest.raises(Exception):
            _ = deployer.generateFunction(verbosityCfg)

        print("\033[92mCode Generation test ended, failed as expected!\033[0m")
        sys.exit(0)
    else:

        _ = deployer.prepare(verbosityCfg)

        # Offset the input and output values if signprop
        if signProp:
            test_inputs = [value - inputOffsets[f"input_{i}"] for i, value in enumerate(test_inputs)]

            for i, values in enumerate(test_outputs):
                buffer = deployer.ctxt.lookup(f"output_{i}")
                if buffer._type.referencedType.typeName == "float32_t":
                    continue
                if not buffer._signed:
                    values -= buffer.nLevels // 2

        generateTestNetwork(deployer, test_inputs, test_outputs, args.dumpdir, verbosityCfg)
