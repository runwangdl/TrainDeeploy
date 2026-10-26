#!/usr/bin/env bash
# exp21c: the probe edits must leave DEFAULT builds byte-identical. Compile the HEAD version and the edited version of
# NE16DualPrep.c / NE16DualConv2in.c with the exact flags of a default (no-probe) GAP9_w_NE16 build directory and
# compare the object code (objdump -d, i.e. instructions; the .o files differ only in their debug/line info, the source
# lines moved). Also with -D NE16DUAL_PROBE (exp11 probe) to show that probe build is unchanged too. Run in deeploy_gap9.
set -e
B=/app/Deeploy/DeeployTest/TEST_GAP9_W_NE16/build_x21c_ne_plain/TargetLibraries/GAP9
E=/app/Deeploy/DeeployTest/experiments/deliverable/exp21_remaining_exp/exp21c_Fusion_NE16_Encode_Opt
F=$B/CMakeFiles/deeploygap9.dir/flags.make
DEFS=$(grep '^C_DEFINES' $F | cut -d= -f2-); INCS=$(grep '^C_INCLUDES' $F | cut -d= -f2-); FL=$(grep '^C_FLAGS' $F | cut -d= -f2-)
CC=/app/install/gcc/gap9/bin/riscv32-unknown-elf-gcc; OD=/app/install/gcc/gap9/bin/riscv32-unknown-elf-objdump
T=/tmp/x21c_ident; mkdir -p $T
for f in NE16DualPrep NE16DualConv2in; do
  for extra in "" "-DNE16DUAL_PROBE"; do
    eval $CC $DEFS $INCS $FL -O3 $extra -c $E/identity_check/HEAD_$f.c -o $T/head.o
    eval $CC $DEFS $INCS $FL -O3 $extra -c /app/Deeploy/TargetLibraries/GAP9/src/$f.c -o $T/new.o
    $OD -d $T/head.o | tail -n +4 > $T/head.s; $OD -d $T/new.o | tail -n +4 > $T/new.s
    if cmp -s $T/head.s $T/new.s; then r=IDENTICAL; else r=DIFFERENT; fi
    echo "$f ${extra:-(default)}: $r ($(wc -l < $T/new.s) disassembly lines)"
  done
done
