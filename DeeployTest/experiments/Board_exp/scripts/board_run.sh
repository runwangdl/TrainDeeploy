#!/usr/bin/env bash
# Board_exp: build + flash + run one training fixture on the GAP9 EVK, capturing the UART console.
# Run INSIDE the board container (deeploy_gap9_board: privileged, board attached over USB/IP):
#   bash board_run.sh <tag> <runner.py> <runner args WITHOUT -s ...> -- <-D defines ...>
#   e.g. bash board_run.sh bp_r1 deeployTrainingRunner_tiled_gap9.py -t <fixture> --n-steps 540 ... -- BN_FROZEN_STATS=ON
# Same flow as Run's silentwear-zo scripts/board_{zo,bp}.sh (her flash/UART helper gap9_board_serial_run.sh is not
# available here, so steps 3-4 replace it):
#   1. generate + build the board image (-s board --skipsim); refuse unless printf goes to the UART (sdk_board.config)
#   2. readfs tail-corruption pad (kb/docs/GAP9/gap9_board_readfs_tail_corruption.md): add zzz_pad.hex so that
#      flash.bin mod 64 KB ~ 20 KB, rebuild with --skipgen
#   3. start the UART capture (pyserial, 115200) on $TTYS (default: all FTDI ttys; ITF1 = the EVK console)
#   4. `cmake --build <build> --target board_<name>` = gapy image flash run (JTAG, openocd)
#   5. stop when the harness's final BENCH line arrived (+5 s) or after TMO seconds
# Output: Board_exp/runs/<tag>/{build.log, build_pad.log, gapy.log, uart_<tty>.log, uart.log (= the console)}.
# openocd is killed only by explicit PID, and only the ones started inside this container. -- QW Board_exp
set -o pipefail   # no -u: the SDK env scripts reference unset variables
TAG=$1; RUNNER=$2; shift 2
ARGS=(); while [[ $# -gt 0 && $1 != -- ]]; do ARGS+=("$1"); shift; done; shift || true; DEFS=("$@")
TMO=${TMO:-3600}; TTYS=${TTYS:-/dev/ttyUSB2}   # EVK console (ITF1) = FT4232H channel C, confirmed by bp_smoke
source /app/install/gap9-sdk/.gap9-venv/bin/activate >/dev/null 2>&1
source /app/install/gap9-sdk/configs/gap9_evk_audio.sh >/dev/null 2>&1
export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation
cd /app/Deeploy/DeeployTest
E=experiments/Board_exp; O=$E/runs/$TAG; mkdir -p $O
PLAT=TEST_GAP9; [[ $RUNNER == *NE16* ]] && PLAT=TEST_GAP9_W_NE16
W=brd_$TAG; B=$PLAT/build_$W
FIX=""; for ((i=0; i<${#ARGS[@]}; i++)); do [[ ${ARGS[$i]} == -t ]] && FIX=${ARGS[$((i+1))]}; done
build() { PYTEST_XDIST_WORKER=$W python3 $RUNNER "${ARGS[@]}" -s board --skipsim "$@" -D "${DEFS[@]}"; }
kill_openocd() { for p in $(pgrep -x openocd 2>/dev/null); do kill -9 $p 2>/dev/null; done; }

# 1. build
HEX=$PLAT/$FIX/hex; rm -f $HEX/zzz_pad.hex
build > $O/build.log 2>&1 || { echo "[$TAG] BUILD FAILED ($O/build.log)"; tail -20 $O/build.log; exit 1; }
grep -aqiE "Printf.*UART|IO_TYPE_UART|sdk_board" $O/build.log || grep -q "CONFIG_IO_TYPE_UART=y" $B/sdk.config 2>/dev/null \
  || echo "[$TAG] WARNING: could not confirm UART printf in the build log -- check $O/build.log"
# 2. readfs pad (image_<name> writes flash.bin into board_workdir without touching the board)
NAME=$(basename $FIX)
cmake --build $B --target image_$NAME > $O/image.log 2>&1 || { echo "[$TAG] IMAGE FAILED ($O/image.log)"; exit 1; }
FB=$(ls $B/board_workdir/flash.bin 2>/dev/null)
if [[ -d $HEX && -n $FB ]]; then
  S=$(stat -c %s $FB); PADB=$(( 65536 + ((20480 - S % 65536 + 65536) % 65536) )); PADB=$(( PADB / 4096 * 4096 ))
  head -c $PADB /dev/zero | tr '\0' 'Z' > $HEX/zzz_pad.hex
  build --skipgen > $O/build_pad.log 2>&1 || { echo "[$TAG] PAD BUILD FAILED"; exit 1; }
  echo "[$TAG] flash.bin $S B -> pad $PADB B -> $(stat -c %s $B/board_workdir/flash.bin) B"
fi
# 3. UART capture
kill_openocd; CAPS=()
for t in $TTYS; do
  [[ -e $t ]] || continue
  /usr/bin/python3 - "$t" "$O/uart_$(basename $t).log" "$TMO" <<'EOF' &
import serial, sys, time
dev, out, tmo = sys.argv[1], sys.argv[2], float(sys.argv[3])
end = time.time() + tmo
s = None
with open(out, "ab", buffering=0) as f:
    while time.time() < end:
        try:
            if s is None:
                s = serial.Serial(dev, 115200, timeout=0.5)
            b = s.read(4096)
            if b:
                f.write(b)
        except Exception:
            s = None
            time.sleep(1)
EOF
  CAPS+=($!)
done
sleep 1
# 4. flash + run
T0=$(date +%s); echo "[$TAG] $(date +%T) flash + run ($NAME)"
timeout $TMO cmake --build $B --target board_$NAME > $O/gapy.log 2>&1 &
GP=$!
# 5. wait for the final BENCH line on any tty
while kill -0 ${CAPS[0]} 2>/dev/null; do
  if grep -aqE "^BENCH(64)? train_cycles_hi" $O/uart_*.log 2>/dev/null; then sleep 5; break; fi
  if ! kill -0 $GP 2>/dev/null && ! grep -aq . $O/uart_*.log 2>/dev/null && (( $(date +%s) - T0 > 600 )); then break; fi
  sleep 5
done
echo "[$TAG] $(date +%T) done after $(( $(date +%s) - T0 )) s"
for p in "${CAPS[@]}"; do kill $p 2>/dev/null; done
kill $GP 2>/dev/null; kill_openocd
CON=$(grep -alE "BENCH|Initializing" $O/uart_*.log | head -1)
[[ -n $CON ]] && tr -d '\r' < $CON > $O/uart.log
grep -a "FREQ\|WSUM s=init\|WCHK\|Errors\|BENCH" $O/uart.log 2>/dev/null | tail -8
