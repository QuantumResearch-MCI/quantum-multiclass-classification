import json
F = 'comparison_classical_vs_quantum.ipynb'
nb = json.load(open(F, encoding='utf-8'))

def code(t): return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': t.strip('\n').splitlines(keepends=True)}
def md(t):   return {'cell_type': 'markdown', 'metadata': {}, 'source': t.strip('\n').splitlines(keepends=True)}
def setsrc(c, t): c['source'] = t.strip('\n').splitlines(keepends=True)
def findcode(m):
    for c in nb['cells']:
        if c['cell_type'] == 'code' and m in ''.join(c['source']): return c
    raise RuntimeError(m)

# (0) judul
nb['cells'][0]['source'] = ['# Komparasi Classical vs Quantum vs Deep Learning']

# (1) deteksi: DL -> paradigma 'Deep Learning'
DETECT = """import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTDIR = '../output_qml/comparison'
os.makedirs(OUTDIR, exist_ok=True)
def savefig(n): plt.savefig(OUTDIR + '/%s.png' % n, dpi=300, bbox_inches='tight')

# AUTO-DETEKSI semua hasil: classical + setiap folder di output_qml (kecuali 'comparison')
PATHS = {'Classical': '../logs/hpo_best.csv'}
for d in sorted(os.listdir('../output_qml')):
    p = '../output_qml/%s/hpo_best.csv' % d
    if d != 'comparison' and os.path.exists(p):
        PATHS[d] = p

def _paradigm(tag):
    if tag == 'Classical': return 'Classical'
    if tag == 'DL':        return 'Deep Learning'   # model dari !DL (MLP, 1D CNN)
    return 'Quantum'

frames = []
for tag, p in PATHS.items():
    d = pd.read_csv(p)
    d['paradigm'] = _paradigm(tag)
    d['group'] = tag
    frames.append(d)
allm = pd.concat(frames, ignore_index=True)
METRICS = ['acc', 'precision', 'recall', 'f1', 'mcc', 'auroc', 'prauc', 'composite']
_pc = allm.paradigm.value_counts().to_dict()
print('Grup ditemukan:', list(PATHS.keys()))
print('Total model:', len(allm), '|', ' | '.join('%s: %d' % (k, _pc.get(k, 0))
      for k in ['Classical', 'Quantum', 'Deep Learning']))"""
setsrc(findcode("PATHS = {'Classical'"), DETECT)

# (2) sisipkan headline 3-arah setelah cell headline classical-vs-quantum (cell 5)
hidx = next(i for i, c in enumerate(nb['cells'])
            if c['cell_type'] == 'code' and "bc = allm[allm.paradigm == 'Classical'].nlargest(1" in ''.join(c['source']))
THREE = """# Best per-paradigma: Classical vs Quantum vs Deep Learning (urut composite)
has_dl = (allm.paradigm == 'Deep Learning').any()
bc = allm[allm.paradigm == 'Classical'].nlargest(1, 'composite').iloc[0]
bq = allm[allm.paradigm == 'Quantum'].nlargest(1, 'composite').iloc[0]
rows3 = [bc, bq]
if has_dl:
    bd = allm[allm.paradigm == 'Deep Learning'].nlargest(1, 'composite').iloc[0]
    rows3.append(bd)
three = pd.DataFrame(rows3)[['paradigm', 'model', 'config'] + METRICS]
print(three.round(4).to_string(index=False))
three.to_csv(OUTDIR + '/comparison_three_way_best.csv', index=False)

# Pemenang per metrik (3 paradigma)
print('\\nPemenang per metrik:')
for m in METRICS:
    vals = {'Classical': bc[m], 'Quantum': bq[m]}
    if has_dl: vals['Deep Learning'] = bd[m]
    win = max(vals, key=vals.get)
    extra = (' DL=%.4f' % bd[m]) if has_dl else ''
    print('  %-10s C=%.4f Q=%.4f%s -> %s' % (m, bc[m], bq[m], extra, win))

# Grouped bar 3-arah
x = np.arange(len(METRICS)); w = 0.27 if has_dl else 0.38
fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(x - (w if has_dl else w/2), [bc[m] for m in METRICS], w, label='Classical: %s' % bc['model'], color='#4C72B0')
ax.bar(x + (0 if has_dl else w/2),  [bq[m] for m in METRICS], w, label='Quantum: %s' % bq['model'], color='#DD8452')
if has_dl:
    ax.bar(x + w, [bd[m] for m in METRICS], w, label='Deep Learning: %s' % bd['model'], color='#55A868')
ax.set_xticks(x); ax.set_xticklabels([m.upper() for m in METRICS], rotation=0, fontsize=9)
ax.set_ylim(0, 1.02); ax.set_ylabel('Skor'); ax.grid(alpha=0.3, axis='y')
ax.set_title('Best per Paradigma — Classical vs Quantum vs Deep Learning (multi-metrik)')
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout(); savefig('07_best_three_way'); plt.show()"""
nb['cells'][hidx+1:hidx+1] = [md('## 2b. Headline 3 paradigma: Classical vs Quantum vs Deep Learning'), code(THREE)]

json.dump(nb, open(F, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
codes = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
for i, s in enumerate(codes): compile(s, '<c%d>' % i, 'exec')
print('OK patched %s | %d cells (compile lolos)' % (F, len(nb['cells'])))
