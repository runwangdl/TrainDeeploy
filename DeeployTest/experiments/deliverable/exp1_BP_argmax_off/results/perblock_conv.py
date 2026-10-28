import re
from collections import defaultdict
L=open("profiletiling.log").read().splitlines()
node=defaultdict(lambda:[0,0,0])
rx=re.compile(r'^\[([^\]]+)\]\[[SD]B\]\[\d+ ops\]\[Tile \d+\] (Pre-Kernel|Kernel|Post-Kernel)\s*:\s*(\d+) cycles')
for ln in L:
    m=rx.match(ln)
    if not m: continue
    idx={"Pre-Kernel":0,"Kernel":1,"Post-Kernel":2}[m.group(2)]
    node[m.group(1)][idx]+=int(m.group(3))
def blk(t):
    m=re.search(r'blocks_(\d+)_',t); return int(m.group(1)) if m else -1
fwdC=defaultdict(int); bwdC=defaultdict(int)
for tag,(p,k,po) in node.items():
    b=blk(tag); tot=p+k+po
    # forward conv: has _Conv_Conv and NOT grad and NOT transpose
    isgrad = ("grad" in tag.lower() or "Grad" in tag)
    istrans= "transpose" in tag.lower()
    if istrans: continue
    if "_Conv_Conv" in tag and not isgrad:
        fwdC[b]+=tot
    if ("ConvGrad" in tag or "GradConv" in tag) and not istrans:
        bwdC[b]+=tot
print(f"{'block':6} {'fwd Conv':>12} {'ConvGrad':>12} {'bwd/fwd':>8}   note")
notes={0:'first layer: dW only (no dX)',1:'dX+dW',2:'dX+dW',3:'dX+dW',4:'dX+dW'}
tf=tb=0
for b in range(5):
    f=fwdC[b]; g=bwdC[b]; tf+=f; tb+=g
    r=(g/f) if f else 0
    print(f"  B{b:<4} {f:>12,} {g:>12,} {r:>7.2f}x   {notes[b]}")
print(f"  {'TOT':<4} {tf:>12,} {tb:>12,} {tb/tf:>7.2f}x")
# shapes for context
print("\nconv shapes: B0 [1,4] 1->8 in[14x700] | B1 [1,16] 8->16 | B2 [1,8] 16->16 | B3 [7,1] 16->32 | B4 [7,1] 32->32")
