#!/usr/bin/env bash
# exp21e: device ZO 2-step run, STRICT fp32 everywhere relevant (-fno-fast-math -ffp-contract=off on the generated
# network + every fp32 kernel source the SpeechNet ZO/BP graphs touch). Worker x21e_zo_strict (own build dir).
# Waits for the default arm (same generation dir) to finish first.
EXPC=/app/ETH/TrainDeeploy/DeeployTest/experiments/deliverable/exp21_remaining_exp/exp21e_numerical_drift
H=/Users/qiwenwu/ETH/TrainDeeploy/DeeployTest/experiments/deliverable/exp21_remaining_exp/exp21e_numerical_drift
while docker exec traindeeploy kill -0 65051 >/dev/null 2>&1; do sleep 20; done   # 65051 = default-arm runner PID (this session)
FILES="Convolution_fp32.c;Conv4x2.c;BatchNorm.c;Relu.c;MaxPool.c;GlobalAveragePool.c;Gemm.c;Matmul.c;Softmax.c;RandomNoise.c;ZORuntime.c;AvgPool.c;ConvGrad.c;PWConvGrad.c;DWConvGrad.c;DWConvolution_fp32.c;Util.c"
docker exec traindeeploy bash -lc "cd /app/ETH/TrainDeeploy/DeeployTest && PYTEST_XDIST_WORKER=x21e_zo_strict python3 -u deeployMezoRunner_tiled_siracusa.py \
    -t Tests/Models/Training/SpeechNet/speechnet_zo_train_b1_fold3_ref --optimizer-dir Tests/Models/Training/SpeechNet/speechnet_zo_update_b1_fold3_ref \
    --n-steps 2 --n-accum 4 --num-data-inputs 2 --eps 0.01 --lr 3e-6 --q 1 --seed 42 \
    --l1 128000 --l2 2000000 --defaultMemLevel L2 --memAllocStrategy MiniMalloc --searchStrategy random-max --cores 8 \
    -D BN_FROZEN_STATS=ON DUMP_WEIGHTS=ON DEEPLOY_STRICT_FP32=ON 'DEEPLOY_STRICT_FP32_FILES=$FILES'" > $H/logs/dev_zo_strict_2steps.log 2>&1
