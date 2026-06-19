"""
REUP-HE — Re-uploading Hardware-Efficient feature map (proposed).

Dirancang khusus untuk Projected Quantum Kernel (PQK), yang memproyeksikan state ke
ekspektasi Pauli 1-qubit <X_k>, <Y_k>, <Z_k>. Feature map IQP standar (H + RZ + RZZ)
hanya membuat <Z> informatif sehingga 2/3 fitur proyeksi PQK terbuang. REUP-HE memperbaiki
ini dengan:

  1. Rotasi MULTI-BASIS (RY lalu RZ) -> <X>, <Y>, <Z> ketiganya membawa informasi.
  2. Entanglement ZZ data-dependent (full/linear/circular) -> interaksi antar-fitur
     mengalir ke marginal 1-qubit lewat entanglement.
  3. BANDWIDTH `c` (skala input) -> mencegah kernel concentration & menyetel ketajaman
     (Shaydulin & Wild, 2022).
  4. DATA RE-UPLOADING `reps` lapisan -> menaikkan ekspresivitas (Perez-Salinas et al., 2020).

Hyperparameter circuit (reps, bandwidth) di-fix agar SEARCH SPACE tetap sama dengan model
quantum lain (yang dituning: C, PCA/n_qubits; lambda_ & gamma statis). `lambda_` = skala
entanglement (dipakai sama seperti feature map lain).
"""

import itertools
from qiskit import QuantumCircuit


class ReuploadCircuits:
    def __init__(self, n_qubits, lambda_=1.0, reps=2, bandwidth=0.5):
        self.n_qubits = n_qubits
        self.lambda_ = lambda_
        self.reps = reps
        self.c = bandwidth

    def _build(self, params, topology):
        n, c, lam = self.n_qubits, self.c, self.lambda_
        if topology == 'full':
            pairs = list(itertools.combinations(range(n), 2))
        elif topology == 'linear':
            pairs = [(i, i + 1) for i in range(n - 1)]
        else:  # circular
            pairs = [(i, (i + 1) % n) for i in range(n)] if n > 1 else []

        qc = QuantumCircuit(n)
        for _ in range(self.reps):
            # encoding multi-basis (bandwidth-scaled) -> aktifkan <X>,<Y>,<Z>
            for i in range(n):
                qc.ry(c * params[i], i)
                qc.rz(c * params[i], i)
            # entanglement ZZ data-dependent
            for i, j in pairs:
                qc.rzz(lam * c * params[i] * params[j], i, j)
        return qc

    def reup_full(self, params):
        return self._build(params, 'full')

    def reup_linear(self, params):
        return self._build(params, 'linear')

    def reup_circular(self, params):
        return self._build(params, 'circular')

    # ===== REUP-HE v2: rotasi multi-basis ENTANGLED (gaya 'star') di banyak pasangan
    # + suku interaksi x_i*x_j di basis Y + re-uploading + bandwidth. Lebih ekspresif. =====
    def _build_ent(self, params, topology):
        n, c, lam = self.n_qubits, self.c, self.lambda_
        if topology == 'full':
            pairs = list(itertools.combinations(range(n), 2))
        elif topology == 'linear':
            pairs = [(i, i + 1) for i in range(n - 1)]
        else:  # circular
            pairs = [(i, (i + 1) % n) for i in range(n)] if n > 1 else []

        qc = QuantumCircuit(n)
        for _ in range(self.reps):
            for i in range(n):
                qc.h(i)
            for i, j in pairs:
                qc.cx(i, j)
                qc.rx(lam * c * params[i], j)               # X (linear)
                qc.ry(lam * c * params[i] * params[j], j)   # Y (interaksi x_i*x_j)
                qc.rz(lam * c * params[j], j)               # Z (linear)
                qc.cx(i, j)
        return qc

    def reup_ent_full(self, params):
        return self._build_ent(params, 'full')

    def reup_ent_linear(self, params):
        return self._build_ent(params, 'linear')

    def reup_ent_circular(self, params):
        return self._build_ent(params, 'circular')

    # ===== REUP-HE v3 (lite): turunan 'star' (entanglement RINGAN: 1 hub/rep, hub berputar)
    # + multi-basis RX/RY/RZ entangled + suku interaksi + bandwidth + re-uploading.
    # Menghindari concentration (beda dgn v2 full/circular yg entanglement-nya padat). =====
    def reup_lite(self, params):
        n, c, lam = self.n_qubits, self.c, self.lambda_
        qc = QuantumCircuit(n)
        for rep in range(self.reps):
            for i in range(n):
                qc.h(i)
                qc.rz(c * params[i], i)
            hub = rep % n  # hub berputar tiap layer -> entanglement ringan tapi menyebar
            for j in range(n):
                if j == hub:
                    continue
                qc.cx(hub, j)
                qc.rx(lam * c * params[hub], j)
                qc.ry(lam * c * params[j], j)
                qc.rz(lam * c * params[hub] * params[j], j)
                qc.cx(hub, j)
        return qc

    # ===== STAR-v2: UPGRADE 'star' (juara quantum), tetap RINGAN (1 hub) tapi:
    #   (1) encode awal multi-basis lebih kaya (H + RY + RZ) -> <X>,<Y>,<Z> lebih informatif
    #   (2) bandwidth `c` pada semua sudut -> cegah concentration & selaras temuan gamma rendah
    #   (3) entanglement tetap 1 hub (ringan) — yang membuat 'star' menang. =====
    def star_v2(self, params):
        n, c, lam = self.n_qubits, self.c, self.lambda_
        qc = QuantumCircuit(n)
        for i in range(n):
            qc.h(i)
            qc.ry(c * params[i], i)
            qc.rz(c * params[i], i)
        hub = 0
        for j in range(n):
            if j == hub:
                continue
            qc.cx(hub, j)
            qc.rx(c * lam * params[hub], j)
            qc.ry(c * lam * params[hub] * params[j], j)   # interaksi x_hub*x_j di Y
            qc.rz(c * lam * params[j], j)
            qc.cx(hub, j)
        return qc
