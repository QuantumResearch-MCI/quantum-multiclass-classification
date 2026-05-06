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

class QuantumKernelEstimator:
    
    def __init__(self, n_qubits, lambda_=1.0, kernel='full', n_measurements=1024):
        self.kernel = kernel
        self.n_measurements = n_measurements
        self.backend = None
        self.n_qubits = n_qubits
        self.lambda_ = lambda_
        self._qkernel = None
        
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
    
    def build_quantum_kernel(self, n_features, use_hardware=False):
        self._feature_map = self._build_feature_map(n_features)

        # create FQK
        if use_hardware:
            # aer_sim = AerSimulator()
            service = QiskitRuntimeService()
            self.backend = service.least_busy(simulator=False, operational=True)
            
            print("Using backend:", self.backend.name)

            pm = generate_preset_pass_manager(backend=self.backend)
            isa_feature_map = pm.run(self._feature_map)

            self.sampler = SamplerV2(mode=self.backend)
            self.sampler.options.default_shots = int(self.n_measurements)
            
            # create fidelity implementation
            fidelity = ComputeUncompute(sampler=self.sampler, transpiler=pm)
            self._qkernel = FidelityQuantumKernel(
                feature_map=isa_feature_map, 
                fidelity=fidelity, 
                max_circuits_per_job=300
            )
        else:
            transpiled_feature_map = transpile(self._feature_map, AerSimulator())
            self._qkernel = FidelityStatevectorKernel(feature_map=transpiled_feature_map)

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