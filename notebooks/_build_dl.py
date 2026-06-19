import json, copy

SRC = 'new_processing_QSVC.ipynb'   # kerangka identik (group-CV, composite, viz, cache)
nb = json.load(open(SRC, encoding='utf-8'))

def find(m):
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] == 'code' and m in ''.join(c['source']): return i
    raise RuntimeError(m)
def findmd(m):
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] == 'markdown' and m in ''.join(c['source']): return i
    raise RuntimeError(m)
def setsrc(c, t): c['source'] = t.strip('\n').splitlines(keepends=True)

OUTDIR = '../output_qml/DL'

IMPORTS = """import os, sys, datetime, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('..'))

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, ParameterGrid
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             matthews_corrcoef, roc_auc_score, average_precision_score)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

OUTDIR = '%s'
os.makedirs(OUTDIR, exist_ok=True)
def savefig(_name):
    plt.savefig(OUTDIR + '/%%s.png' %% _name, dpi=300, bbox_inches='tight')
""" % OUTDIR

CONFIG = """dataset_path = '../dataset/datasets.csv'
feature_cols = ['MQ3', 'TGS822', 'TGS2602', 'MQ5', 'MQ138', 'TGS2620',
                'TGS813', 'TGS2600', 'TGS2611', 'TGS2603', 'Humidity', 'Celsius']
target_col = 'Kategori'
group_col  = 'Sampling_ID'

CV_MODE       = 'group'        # group-aware anti-leakage (sama dgn semua model lain)
n_splits      = 5
USE_PCA       = True
random_states = [42]
select_by     = 'composite'
DRAW_LEARNING_CURVES = False   # DL punya kurva per-epoch sendiri; learning_curve sklearn = retrain Keras berkali2 (mahal)

PCA_COMPONENTS = [3, 5, 8, 12]   # PCA dituning, sama dgn model lain

# Model DL (arsitektur & hyperparameter dari !DL best). Hanya PCA yang disweep.
SEARCH_SPACES = {'MLP': {}, '1D-CNN': {}}
"""

BASE = """RS = 42

class _KerasBase(BaseEstimator, ClassifierMixin):
    _conv = False
    def _reshape(self, X):
        X = np.asarray(X, dtype='float32')
        return X[..., np.newaxis] if self._conv else X
    def fit(self, X, y):
        Xr = self._reshape(X); y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)
        cw = compute_class_weight('balanced', classes=self.classes_, y=y)   # tangani imbalance (sama spt SVC balanced)
        cwd = {int(c): float(w) for c, w in zip(self.classes_, cw)}
        Xtr, Xva, ytr, yva = train_test_split(Xr, y, test_size=0.15, random_state=self.random_state, stratify=y)
        keras.backend.clear_session(); keras.utils.set_random_seed(self.random_state)
        self.model_ = self._build(Xr.shape[1], len(self.classes_))
        es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)
        self.model_.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=self.epochs,
                        batch_size=self.batch_size, class_weight=cwd, verbose=0, callbacks=[es])
        return self
    def predict_proba(self, X):
        return self.model_.predict(self._reshape(X), verbose=0)
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

class KerasMLP(_KerasBase):
    _conv = False
    def __init__(self, hidden_units=(128, 64, 32), dropout=0.2, learning_rate=0.001,
                 batch_size=64, epochs=100, l2=1e-4, random_state=42):
        self.hidden_units = hidden_units; self.dropout = dropout; self.learning_rate = learning_rate
        self.batch_size = batch_size; self.epochs = epochs; self.l2 = l2; self.random_state = random_state
    def _build(self, input_dim, n_classes):
        inp = keras.Input(shape=(input_dim,)); x = inp
        for h in self.hidden_units:
            x = layers.Dense(h, activation='relu', kernel_initializer='he_normal',
                             kernel_regularizer=regularizers.l2(self.l2))(x)
            x = layers.BatchNormalization()(x); x = layers.Dropout(self.dropout)(x)
        out = layers.Dense(n_classes, activation='softmax')(x)
        m = keras.Model(inp, out)
        m.compile(optimizer=keras.optimizers.Adam(self.learning_rate),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return m

class KerasCNN1D(_KerasBase):
    _conv = True
    def __init__(self, filters=(64, 128), kernel_size=3, dropout=0.2, learning_rate=0.001,
                 batch_size=64, epochs=100, l2=1e-4, random_state=42):
        self.filters = filters; self.kernel_size = kernel_size; self.dropout = dropout
        self.learning_rate = learning_rate; self.batch_size = batch_size; self.epochs = epochs
        self.l2 = l2; self.random_state = random_state
    def _build(self, input_len, n_classes):
        inp = keras.Input(shape=(input_len, 1)); x = inp
        for f in self.filters:
            x = layers.Conv1D(f, self.kernel_size, padding='same', activation='relu',
                              kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(self.l2))(x)
            x = layers.BatchNormalization()(x)
            if x.shape[1] is not None and x.shape[1] > 1:
                x = layers.MaxPooling1D(pool_size=2)(x)
            x = layers.Dropout(self.dropout)(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(self.l2))(x)
        x = layers.Dropout(self.dropout)(x)
        out = layers.Dense(n_classes, activation='softmax')(x)
        m = keras.Model(inp, out)
        m.compile(optimizer=keras.optimizers.Adam(self.learning_rate),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return m

BASE_MODELS = {'MLP': KerasMLP(random_state=RS), '1D-CNN': KerasCNN1D(random_state=RS)}

def build_grid(name):
    grid = {'model__' + k: v for k, v in SEARCH_SPACES[name].items()}
    if USE_PCA:
        grid['pca__n_components'] = PCA_COMPONENTS
    return grid

specs = [(name, BASE_MODELS[name], build_grid(name)) for name in BASE_MODELS]
print('=== SEARCH SPACE (Deep Learning) ===')
for name in BASE_MODELS:
    print('  %-8s | %d config (PCA disweep)' % (name, len(list(ParameterGrid(build_grid(name))))))
total = sum(len(list(ParameterGrid(build_grid(n)))) for n in BASE_MODELS)
print('TOTAL: %d config x %d fold = %d fit (Keras, group-CV, class_weight balanced)' % (total, n_splits, total * n_splits))"""

setsrc(nb['cells'][findmd('HPO Quantum')],
       '# HPO Deep Learning (MLP & 1D CNN) | Anti-Leakage\n\n'
       'Model DL dari `!DL.ipynb` (MLP & 1D CNN, Keras) dievaluasi dgn **protokol IDENTIK** model lain: '
       'group-CV (`StratifiedGroupKFold`, anti-leakage), `Scaler→PCA` (PCA {3,5,8,12} disweep), '
       '**composite minority-aware**, `class_weight=balanced` (tanpa resampling, 5 kelas penuh). '
       'Output ke `output_qml/DL/`. Agar adil dibandingkan ke classical & quantum.')
setsrc(nb['cells'][find('from catboost import CatBoostClassifier') if False else find('matplotlib.pyplot as plt')], IMPORTS)
setsrc(nb['cells'][find('SEARCH_SPACES = {')], CONFIG)
setsrc(nb['cells'][find('BASE_MODELS = {')], BASE)

for c in nb['cells']:
    if c['cell_type'] == 'code':
        c['source'] = ''.join(c['source']).replace('../output_qml/QSVC', OUTDIR).splitlines(keepends=True)

json.dump(nb, open('new_processing_DL.ipynb', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
codes = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
for i, s in enumerate(codes): compile(s, '<c%d>' % i, 'exec')
print('OK -> new_processing_DL.ipynb | %d cells, %d code (compile lolos)' % (len(nb['cells']), len(codes)))
