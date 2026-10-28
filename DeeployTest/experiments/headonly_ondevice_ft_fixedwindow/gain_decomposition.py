"""Decompose the head-only FT batch-2 gain: base checkpoint (inter_session vs
inter_session_ft) x windowing (center vs onset), plus draw-variance for the new config.
Head-only + BN-fold sim = eval-mode BN (calibrated bit-exact to device for this path),
train only fc, sum-accumulation SGD (n_accum 4, lr 0.01, 40 epochs, fixed order)."""
import numpy as np, torch, sys, types
import torch.nn as nn
import pandas as pd
sys.path.insert(0, '/app/Onnx4Deeploy')
from onnx4deeploy.models.pytorch_models.speechnet.speechnet import SpeechNetDeploy
from onnx4deeploy.data.silent_wear_datasource import EMG_FILT_COLS

DATA = '/app/SilentWear/SilentWear_data/data_raw_and_filt/S01/vocalized'
CKPTS = {
    'inter_session':    '/app/SilentWear/SilentWear/artifacts/models/inter_session/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt',
    'inter_session_ft': '/app/SilentWear/SilentWear/artifacts/models/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt',
}
WIN = 700

def patched(self, x):
    for b in self.blocks: x = b(x)
    x = self.global_pool(x); x = x.reshape(x.shape[0], self._fc_in); return self.fc(x)

def windows(batch, anchor):
    df = pd.read_hdf(f'{DATA}/sess_3_batch_{batch}.h5', key='emg')
    emg = df[EMG_FILT_COLS].values.astype(np.float32); lab = df['Label_int'].values
    chg = np.where(np.diff(lab) != 0)[0] + 1
    starts = np.concatenate([[0], chg]); ends = np.concatenate([chg, [len(lab)]])
    X, y = [], []
    for s, e in zip(starts, ends):
        if anchor == 'onset':
            if s + WIN > len(emg): continue
            w = emg[s:s+WIN]
        else:  # center
            if e - s < WIN: continue
            off = (e - s - WIN) // 2; w = emg[s+off:s+off+WIN]
        X.append(w.T[None, None]); y.append(int(lab[s]))
    return X, np.array(y)

def ds_rest(X, y, seed=42):
    from collections import Counter
    c = Counter(y.tolist()); mn = min(v for k, v in c.items() if k != 0)
    ri = np.where(y == 0)[0]
    if len(ri) > mn:
        keep_r = np.random.RandomState(seed).choice(ri, mn, replace=False)
        keep = np.array(sorted(list(np.where(y != 0)[0]) + list(keep_r)))
        X = [X[i] for i in keep]; y = y[keep]
    return X, y

def draw30(X, y, seed):
    rng = np.random.default_rng(seed); idx = []
    for c in sorted(set(y.tolist())):
        idx += list(rng.choice(np.where(y == c)[0], 6, replace=False))
    idx = np.array(sorted(idx))
    return [X[i] for i in idx], y[idx]

def mk(ckpt):
    sd = torch.load(ckpt, map_location='cpu', weights_only=False); sd = sd.get('model_state_dict', sd)
    m = SpeechNetDeploy(num_channels=14, time_steps=700, num_classes=9); m.load_state_dict(sd)
    m.forward = types.MethodType(patched, m); return m

def bacc(m, X, y):
    m.eval(); cls = sorted(set(y.tolist()))
    with torch.no_grad():
        p = np.array([m(torch.from_numpy(X[i].astype(np.float32))).argmax(1).item() for i in range(len(y))])
    return 100*float(np.mean([(p[y==c]==c).mean() for c in cls]))

def ft_head(m, X, y, lr=0.01, nacc=4, epochs=40):
    m.eval()  # freeze BN (folded-equivalent)
    for p in m.parameters(): p.requires_grad = False
    m.fc.weight.requires_grad = True; m.fc.bias.requires_grad = True
    opt = torch.optim.SGD([m.fc.weight, m.fc.bias], lr=lr); crit = nn.CrossEntropyLoss()
    N = len(X)
    for e in range(epochs):
        opt.zero_grad(); c = 0
        for j in range(N):
            crit(m(torch.from_numpy(X[j].astype(np.float32))), torch.tensor([int(y[j])])).backward(); c += 1
            if c % nacc == 0: opt.step(); opt.zero_grad()
        if c % nacc: opt.step(); opt.zero_grad()
    return m

print('base            window   b2 zero-shot   b2 after head-FT   gain')
for base, ckpt in CKPTS.items():
    for anchor in ('center', 'onset'):
        Xb2, yb2 = ds_rest(*windows(2, anchor))
        zs = bacc(mk(ckpt), Xb2, yb2)
        Xb1, yb1 = ds_rest(*windows(1, anchor)); Xtr, ytr = draw30(Xb1, yb1, 42)
        m = ft_head(mk(ckpt), Xtr, ytr); ft = bacc(m, Xb2, yb2)
        print('%-16s %-7s  %6.2f%%        %6.2f%%          %+5.2f pp' % (base, anchor, zs, ft, ft-zs))

# draw variance for the NEW config (onset + inter_session_ft)
Xb2, yb2 = ds_rest(*windows(2, 'onset')); zs = bacc(mk(CKPTS['inter_session_ft']), Xb2, yb2)
Xb1, yb1 = ds_rest(*windows(1, 'onset'))
gains = []
for s in range(10):
    Xtr, ytr = draw30(Xb1, yb1, 1000+s)
    ft = bacc(ft_head(mk(CKPTS['inter_session_ft']), Xtr, ytr), Xb2, yb2)
    gains.append(ft - zs)
g = np.array(gains)
print('\nNEW config (onset + inter_session_ft) b1->b2 gain over 10 draws: mean %+.2f, std %.2f, range %+.2f..%+.2f (zs=%.2f%%)'
      % (g.mean(), g.std(), g.min(), g.max(), zs))
