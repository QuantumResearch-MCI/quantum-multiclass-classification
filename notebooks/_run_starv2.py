import json, warnings, matplotlib, time
warnings.filterwarnings('ignore'); matplotlib.use('Agg')
t0 = time.time()
nb = json.load(open('new_proposed_star-v2.ipynb', encoding='utf-8'))
g = {'__name__': '__main__'}
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        try:
            exec(''.join(c['source']), g)
        except Exception as e:
            print('ERROR cell', i, type(e).__name__, e); raise
        import matplotlib.pyplot as plt; plt.close('all')
print('DONE in %.0f s' % (time.time() - t0))
import pandas as pd
b = pd.read_csv('../output_qml/star-v2/hpo_best.csv').sort_values('composite', ascending=False)
print(b[['model', 'config', 'acc', 'precision', 'recall', 'f1', 'mcc', 'auroc', 'prauc', 'composite']].round(4).to_string(index=False))
