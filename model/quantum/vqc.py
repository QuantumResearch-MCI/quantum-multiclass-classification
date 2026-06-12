import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from qiskit.circuit.library import RealAmplitudes, EfficientSU2
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA, SPSA, ADAM, L_BFGS_B
from qiskit_machine_learning.utils import algorithm_globals
from .estimator import QuantumKernelEstimator

# Param yang dipegang langsung oleh wrapper (sisanya diteruskan ke VQC).
_VQC_PARAMS = {
    'n_qubits', 'lambda_', 'kernel', 'mode', 'n_features', 'gamma',
    'random_state', 'ansatz', 'entanglement', 'reps', 'optimizer', 'maxiter',
    'learning_rate', 'n_measurements',
}

_OPTIMIZERS = {
    'cobyla':   lambda mi, lr: COBYLA(maxiter=mi),
    'spsa':     lambda mi, lr: SPSA(maxiter=mi),
    'adam':     lambda mi, lr: ADAM(maxiter=mi, lr=lr),
    'l_bfgs_b': lambda mi, lr: L_BFGS_B(maxiter=mi),
}

_ANSATZE = {
    'real_amplitudes': RealAmplitudes,
    'efficient_su2':   EfficientSU2,
}


class VQCWrapper(BaseEstimator, ClassifierMixin):
    """Variational Quantum Classifier (VQC) yang kompatibel dengan runner.

    Berbeda dari QSVC/QXGB/QCAT (metode KERNEL): VQC melatih sirkuit ansatz
    berparameter via optimizer klasik. Feature map dipinjam dari
    `QuantumKernelEstimator` agar pilihan `kernel` (full/linear/circular/...)
    & `lambda_` konsisten dengan model kernel. Parameter `mode`/`gamma`/
    `n_measurements` diterima demi keseragaman pemanggilan tapi TIDAK dipakai
    (VQC bukan kernel; sampler memakai statevector default Aer).
    """

    def __init__(
        self,
        n_qubits=4,
        lambda_=1.0,
        kernel='full',
        mode='pqk',
        n_features=4,
        gamma=1.0,
        n_measurements=None,
        ansatz='real_amplitudes',
        entanglement='full',
        reps=2,
        optimizer='cobyla',
        maxiter=100,
        learning_rate=0.05,
        random_state=None,
        **vqc_params,
    ):
        self.n_qubits = n_qubits
        self.lambda_ = lambda_
        self.kernel = kernel
        self.mode = mode
        self.n_features = n_features
        self.gamma = gamma
        self.n_measurements = n_measurements
        self.ansatz = ansatz
        self.entanglement = entanglement
        self.reps = reps
        self.optimizer = optimizer
        self.maxiter = maxiter
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.vqc_params = vqc_params

        self.classes_ = None
        self.model_ = None

    def _build_feature_map(self):
        # Pinjam feature map (parameter 'x') dari estimator kernel -> konsisten
        # dengan QSVC dkk. lambda_ menskalakan entanglement IQP.
        est = QuantumKernelEstimator(
            kernel=self.kernel, n_qubits=self.n_qubits, lambda_=self.lambda_,
        )
        return est._build_feature_map(self.n_features)

    def _build_ansatz(self):
        try:
            cls = _ANSATZE[self.ansatz]
        except KeyError:
            raise ValueError(
                f"ansatz '{self.ansatz}' tak dikenal. Pilih {list(_ANSATZE)}.")
        return cls(num_qubits=self.n_qubits, reps=self.reps,
                   entanglement=self.entanglement)

    def _build_optimizer(self):
        try:
            return _OPTIMIZERS[self.optimizer](self.maxiter, self.learning_rate)
        except KeyError:
            raise ValueError(
                f"optimizer '{self.optimizer}' tak dikenal. "
                f"Pilih {list(_OPTIMIZERS)}.")

    def _build_model(self):
        if self.random_state is not None:
            algorithm_globals.random_seed = self.random_state

        feature_map = self._build_feature_map()
        ansatz = self._build_ansatz()

        initial_point = None
        if self.random_state is not None:
            rng = np.random.default_rng(self.random_state)
            initial_point = rng.uniform(0, 2 * np.pi, ansatz.num_parameters)

        # Aer SamplerV2 butuh sirkuit ter-transpile (RealAmplitudes/EfficientSU2
        # & feature map adalah blok high-level) -> beri pass manager.
        pm = generate_preset_pass_manager(backend=AerSimulator(), optimization_level=1)

        return VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=self._build_optimizer(),
            sampler=StatevectorSampler(),
            initial_point=initial_point,
            pass_manager=pm,
            **self.vqc_params,
        )

    def get_params(self, deep=True):
        return {
            'n_qubits': self.n_qubits,
            'lambda_': self.lambda_,
            'kernel': self.kernel,
            'mode': self.mode,
            'n_features': self.n_features,
            'gamma': self.gamma,
            'n_measurements': self.n_measurements,
            'ansatz': self.ansatz,
            'entanglement': self.entanglement,
            'reps': self.reps,
            'optimizer': self.optimizer,
            'maxiter': self.maxiter,
            'learning_rate': self.learning_rate,
            'random_state': self.random_state,
            **self.vqc_params,
        }

    def set_params(self, **params):
        for key, value in params.items():
            if key in _VQC_PARAMS:
                setattr(self, key, value)
            else:
                self.vqc_params[key] = value
        return self

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.model_ = self._build_model()
        self.model_.fit(np.asarray(X, dtype=float), np.asarray(y))
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X, dtype=float))

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, dtype=float))

    def score(self, X, y):
        return self.model_.score(np.asarray(X, dtype=float), np.asarray(y))
