from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from qiskit.circuit import ParameterVector
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit.transpiler import generate_preset_pass_manager
from .circuits import QuantumKernelCircuits
import numpy as np

KERNEL_MODES = ['fsk', 'fqk', 'fqk-hardware']

class QuantumKernelEstimator:
    
    def __init__(self, n_qubits, lambda_=1.0, kernel='full', n_measurements=1024):
        self.kernel = kernel
        self.n_measurements = n_measurements
        self.backend = None
        self.n_qubits = n_qubits
        self.lambda_ = lambda_
        self._qkernel = None
        self._feature_map = None
        
        # Map kernel types to circuit creation functions
        self.qkc = QuantumKernelCircuits(self.n_qubits, self.lambda_)
        self.circuit_creators = {
            'full': self.qkc.create_iqp_full,
            'linear': self.qkc.create_iqp_linear,
            'circular': self.qkc.create_iqp_circular,
            'pauli_x': self.qkc.create_pauli_x,
            'pauli_y': self.qkc.create_pauli_y,
            'pauli_z': self.qkc.create_pauli_z
        }

    def _build_feature_map(self, n_features):
        params = ParameterVector('x', length=n_features)
        creator_fn = self.circuit_creators[self.kernel]
        feature_map = creator_fn(params)
        
        return feature_map
    
    def _build_fsk(self, n_features):
        feature_map = self._build_feature_map(n_features)
        transpiled_map = transpile(feature_map, AerSimulator())
        return FidelityStatevectorKernel(feature_map=transpiled_map)
    
    def _build_fqk(self, n_features):
        feature_map = self._build_feature_map(n_features)
        sampler = SamplerV2()
        sampler.options.default_shots = int(self.n_measurements)
        fidelity = ComputeUncompute(sampler=sampler)
        return FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
    
    def _build_fqk_hardware(self, n_features):
        feature_map = self._build_feature_map(n_features)
        service = QiskitRuntimeService()
        self.backend = service.least_busy(simulator=False, operational=True)
        
        print("Using backend:", self.backend.name)

        pm = generate_preset_pass_manager(backend=self.backend)
        # isa_feature_map = pm.run(feature_map)

        sampler = SamplerV2(mode=self.backend)
        sampler.options.default_shots = int(self.n_measurements)
        
        fidelity = ComputeUncompute(sampler=sampler, transpiler=pm)
        return FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity, max_circuits_per_job=300)

    def build_quantum_kernel(self, n_features, mode='fsk'):
        if mode not in KERNEL_MODES:
            raise ValueError(f"Invalid kernel mode '{mode}'. Must be one of {KERNEL_MODES}.")
        
        self._feature_map = self._build_feature_map(n_features)

        builders = {
            'fsk': self._build_fsk,
            'fqk': self._build_fqk,
            'fqk-hardware': self._build_fqk_hardware
        }

        self._qkernel = builders[mode](n_features)
        return self._qkernel
    
    # ------------------------------------------------------------------
    # Methods below are used by QuantumEnhancedMulticlassSVM (paper model)
    # ------------------------------------------------------------------

    def compute_kernel_matrix(self, X, Y=None):
        """
        Compute kernel matrix using the built Qiskit kernel.
        - If Y is None: computes symmetric K(X, X) for training
        - If Y is provided: computes rectangular K(X, Y) for inference
        
        Must call build_quantum_kernel() before this.
        """
        if self._qkernel is None:
            raise RuntimeError(
                "Quantum kernel not built yet. Call build_quantum_kernel() first."
            )
        
        if Y is None:
            return self._compute_symmetric(X)
        return self._compute_rectangular(X, Y)
    
    def _compute_symmetric(self, X):
        """K(X, X) — symmetric, exploits K[i,j] = K[j,i]."""
        return self._qkernel.evaluate(X, X)

    def _compute_rectangular(self, X, Y):
        """K(X, Y) — rectangular, for test vs train."""
        return self._qkernel.evaluate(X, Y)