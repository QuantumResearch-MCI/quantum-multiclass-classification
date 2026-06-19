import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import json, warnings, matplotlib, time, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore'); matplotlib.use('Agg')
t0 = time.time()
nb = json.load(open('new_processing_DL.ipynb', encoding='utf-8'))
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
b = pd.read_csv('../output_qml/DL/hpo_best.csv').sort_values('composite', ascending=False)
print(b[['model', 'config', 'acc', 'precision', 'recall', 'f1', 'mcc', 'auroc', 'prauc', 'composite']].round(4).to_string(index=False))
