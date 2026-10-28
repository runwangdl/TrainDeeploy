# measured cycles per block (from trace)
fwd_cyc={0:7405374,1:6121186,2:922827,3:175772,4:90059}
bwd_cyc={0:1458515,1:6137453,2:1761296,3:622873,4:352205}
# conv geometry: (in_ch, out_ch, kh, kw, H_out, W_out)
geo={0:(1,8,1,4,14,700),1:(8,16,1,16,14,87),2:(16,16,1,8,14,21),3:(16,32,7,1,14,5),4:(32,32,7,1,14,5)}
dx_skip={0:True,1:False,2:False,3:False,4:False}
FREQ=370e6
def macs_fwd(g):
    ic,oc,kh,kw,H,W=g; return oc*H*W*ic*kh*kw
print(f"{'blk':4}{'fwdMAC':>10}{'fwd cyc':>10}{'f cyc/MAC':>10} | {'bwdMAC(dX+dW)':>14}{'bwd cyc':>10}{'b cyc/MAC':>10} | {'FLOPx':>6}{'effx':>6}{'ratio':>7}")
tot_mac=0; tot_cyc=0
for b in range(5):
    fm=macs_fwd(geo[b]); fc=fwd_cyc[b]
    bm = fm*(1 if dx_skip[b] else 2)      # dW (=fwd) + dX (=fwd) unless skipped
    bc=bwd_cyc[b]
    fcpm=fc/fm; bcpm=bc/bm
    flopx=bm/fm; effx=bcpm/fcpm; ratio=bc/fc
    print(f"B{b:<3}{fm:>10,}{fc:>10,}{fcpm:>10.2f} | {bm:>14,}{bc:>10,}{bcpm:>10.2f} | {flopx:>5.0f}x{effx:>5.2f}x{ratio:>6.2f}x")
    tot_mac+=fm+bm; tot_cyc+=fc+bc
# total FLOPs/s over the whole step (use BENCH not just conv-cycles for wall time)
BENCH=34084097
flops=2*tot_mac
print(f"\nconv MACs/step (fwd+bwd) = {tot_mac:,}  -> FLOPs = {flops:,} = {flops/1e6:.1f} MFLOP")
t=BENCH/FREQ
print(f"step wall @370MHz = {BENCH:,} cyc / {FREQ/1e6:.0f}MHz = {t*1000:.1f} ms")
print(f"effective conv throughput = {flops/t/1e9:.3f} GFLOP/s")
peak=8*2*FREQ  # 8 cores x 2 FLOP/cyc (fp32 FMA)
print(f"fp32 peak (8 cores x FMA @370MHz) = {peak/1e9:.2f} GFLOP/s  -> utilization = {100*(flops/t)/peak:.1f}%")
# ROUND
print("\n=== ONE BP FINE-TUNING ROUND (exp4 config: 40 epochs, 54 win, n_accum 4 -> 540 steps) ===")
fb=BENCH; opt=61480
per_step=4*fb+opt
round_cyc=540*per_step
print(f"per fwd+bwd (n_accum 1)  = {fb:,} cyc")
print(f"per device step (4x fb + opt) = {per_step:,} cyc")
print(f"round = 540 steps = {round_cyc:,} cyc = {round_cyc/1e9:.2f} G")
for f in [370e6,240e6]:
    print(f"   @ {f/1e6:.0f} MHz : {round_cyc/f:.1f} s = {round_cyc/f/60:.2f} min")
