#!/usr/bin/env bash
# exp21a (origin check): training-network CODE GENERATION ONLY (no build, no GVSoC) of the BP single step
# (speechnet_train_BP_round1, n_steps 1, n_accum 4, --l1 122000 --l2 1300000) under three memory configurations,
# to separate what --defaultMemLevel L3 alone does from what Run's promotion recipe adds:
#   l3_plain   --defaultMemLevel L3                                  (single buffering, no promotion)
#   l3_db      --defaultMemLevel L3 --doublebuffer                   (no promotion)
#   l3_db_prom --defaultMemLevel L3 --doublebuffer --promoteToL2 ... (= Run's board/CI recipe, = evidence/ run)
# Run inside deeploy_gap9:  bash gen_l3_variants.sh     Output: gen/<cfg>/ (full codegen output; the committed copy was
# pruned afterwards to gen.log, mallocs_TrainingNetwork.c.txt = `grep -n "malloc(\|pos_alloc race" TrainingNetwork.c`,
# deeployStates/memory_alloc.html and the MiniMalloc CSVs). -- QW 2026-10-26
set -o pipefail
source /app/install/gap9-sdk/.gap9-venv/bin/activate >/dev/null 2>&1
cd /app/Deeploy/DeeployTest
E=experiments/deliverable/exp21_remaining_exp/exp21a_BP_L2L3Mem/gen
T=../SpeechNet_Test/speechnet_train_BP_round1
COMMON="--n-steps=1 --n-accum=4 --num-data-inputs=2 --cores=8 --l1=122000 --l2=1300000 --memAllocStrategy=MiniMalloc --searchStrategy=random-max --plotMemAlloc"
PROM="--promoteToL2 --promoteToL2Strategy=traffic-per-peak --promoteToL2IncludeActivations --promoteToL2MaxBufferBytes=0 --promoteToL2Headroom=131072"
gen() { local cfg=$1; shift; mkdir -p $E/$cfg
  python3 testMVPTraining.py -d $E/$cfg -t $T -p GAP9 $COMMON "$@" > $E/$cfg/gen.log 2>&1; echo "$cfg rc=$?"; }
gen l3_plain   --defaultMemLevel=L3
gen l3_db      --defaultMemLevel=L3 --doublebuffer
gen l3_db_prom --defaultMemLevel=L3 $PROM --doublebuffer
