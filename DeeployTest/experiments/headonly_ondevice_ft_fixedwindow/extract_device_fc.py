"""Reconstruct the on-device-trained fc weights from the [WDUMP] raw-hex lines and
validate them against the ORT reference (outputs.npz). FPU-free bit-exact reconstruction:
struct.unpack('<f', struct.pack('<I', word)).

wi=0 n=288 -> fc_weight (9,32) row-major ; wi=1 n=9 -> fc_bias (9,)
"""
import re, struct, sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, 'run_train_isft_fw.log')
TR = '/app/ETH/TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet/speechnet_train_head_ep40_isft_fw'
OUT = HERE

def words_to_floats(words):
    return np.array([struct.unpack('<f', struct.pack('<I', int(w, 16)))[0] for w in words], dtype=np.float32)

dumps = {}
for line in open(LOG):
    m = re.match(r'\[WDUMP s=(\d+) wi=(\d+) n=(\d+)\]\s*(.*)', line)
    if m:
        wi = int(m.group(2)); n = int(m.group(3)); words = m.group(4).split()
        assert len(words) == n, f'wi={wi}: expected {n} words, got {len(words)}'
        dumps[wi] = words_to_floats(words)

fc_w = dumps[0].reshape(9, 32)
fc_b = dumps[1].reshape(9)
np.save(os.path.join(OUT, 'device_fc_weight.npy'), fc_w)
np.save(os.path.join(OUT, 'device_fc_bias.npy'), fc_b)
print('reconstructed device fc_weight', fc_w.shape, 'fc_bias', fc_b.shape)

# validate against ORT reference (the on-device run reported Errors: 0 -> should match closely)
zt = np.load(TR + '/outputs.npz')
dw = float(np.max(np.abs(fc_w - zt['fc_weight'])))
db = float(np.max(np.abs(fc_b - zt['fc_bias'])))
print('max|device - ORT| : fc_weight=%.3e  fc_bias=%.3e' % (dw, db))
print('VALID (device == ORT within fp32)' if dw < 1e-4 and db < 1e-4 else 'WARNING: device fc diverges from ORT')
