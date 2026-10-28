import re
from collections import defaultdict
L=open("profiletiling.log").read().splitlines()
# sum Kernel(compute) and Total per node
K=defaultdict(int); T=defaultdict(int)
rx=re.compile(r'^\[([^\]]+)\]\[[SD]B\]\[\d+ ops\]\[Tile \d+\] (Pre-Kernel|Kernel|Post-Kernel)\s*:\s*(\d+) cycles')
for ln in L:
    m=rx.match(ln)
    if not m: continue
    tag,ph,c=m.group(1),m.group(2),int(m.group(3))
    T[tag]+=c
    if ph=="Kernel": K[tag]+=c
def find(block, which):
    # which in {fwd, X, W, B}
    for tag in T:
        if f"blocks_{block}_" not in tag: continue
        if which=="fwd" and tag.endswith("_Conv_Conv_L2"): return tag
        if which=="X" and "ConvGradX" in tag: return tag
        if which=="W" and "ConvGradW" in tag and "transpose" not in tag.lower(): return tag
        if which=="B" and "ConvGradB" in tag: return tag
    return None
# geometry -> forward MACs; dX and dW MACs = fwd MACs each
geo={0:(1,8,1,4,14,700),1:(8,16,1,16,14,87),2:(16,16,1,8,14,21),3:(16,32,7,1,14,5),4:(32,32,7,1,14,5)}
def fmac(g): ic,oc,kh,kw,H,W=g; return oc*H*W*ic*kh*kw
print(f"{'blk':4}{'fwdMAC':>10} | {'Conv cyc':>10}{'c/MAC':>7} | {'dX cyc':>10}{'c/MAC':>7} | {'dW cyc':>10}{'c/MAC':>7} | {'dB cyc':>8} | dX/fwd dW/fwd (dX+dW)/fwd")
for b in range(5):
    fm=fmac(geo[b])
    ft=T[find(b,'fwd')]
    xt=T[find(b,'X')] if find(b,'X') else 0
    wt=T[find(b,'W')] if find(b,'W') else 0
    bt=T[find(b,'B')] if find(b,'B') else 0
    fcm=ft/fm; xcm=(xt/fm) if xt else 0; wcm=wt/fm
    print(f"B{b:<3}{fm:>10,} | {ft:>10,}{fcm:>7.2f} | {xt:>10,}{xcm:>7.2f} | {wt:>10,}{wcm:>7.2f} | {bt:>8,} | {xt/ft:6.2f} {wt/ft:6.2f} {(xt+wt+bt)/ft:6.2f}")
