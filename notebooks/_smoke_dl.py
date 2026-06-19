import json, os, warnings, matplotlib
warnings.filterwarnings('ignore'); matplotlib.use('Agg')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np

nb = json.load(open('new_processing_DL.ipynb', encoding='utf-8'))
codes = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
g = {'__name__': 'smoke'}
for s in codes:
    if 'all_results = []' in s:      # berhenti sebelum eksekusi HPO penuh
        break
    exec(s, g)

X_all, y_all, groups_all = g['X_all'], g['y_all'], g['groups_all']
make_pipe, fold_metrics, composite_score = g['make_pipe'], g['fold_metrics'], g['composite_score']
KerasMLP, KerasCNN1D, make_cv = g['KerasMLP'], g['KerasCNN1D'], g['make_cv']

cv = make_cv(42)
tr, te = next(iter(cv.split(X_all, y_all, groups_all)))
print('fold train=%d test=%d | classes=%s' % (len(tr), len(te), g['classes']))
for nm, mdl in [('MLP', KerasMLP(epochs=3)), ('1D-CNN', KerasCNN1D(epochs=3))]:
    pipe = make_pipe(mdl); pipe.set_params(pca__n_components=5)
    pipe.fit(X_all[tr], y_all[tr])
    proba = pipe.predict_proba(X_all[te])
    pred = np.asarray(pipe.predict(X_all[te])).ravel().astype(int)
    acc, prec, rec, f1, mcc, au, pr = fold_metrics(y_all[te], pred, proba)
    comp = composite_score(f1, mcc, pr)
    print('%-7s proba=%s rowsum~%.3f | Acc=%.3f F1=%.3f MCC=%.3f AUROC=%.3f PRAUC=%.3f Comp=%.3f'
          % (nm, proba.shape, float(proba.sum(1).mean()), acc, f1, mcc, au, pr, comp))
print('SMOKE OK')
