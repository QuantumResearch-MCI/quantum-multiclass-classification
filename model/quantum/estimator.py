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

class QuantumKernelEstimator:
    
    def __init__(self, n_qubits, lambda_=1.0, kernel_type='full', n_measurements=1024):
        self.kernel_type = kernel_type
        self.n_measurements = n_measurements
        self.backend = None
        self.n_qubits = n_qubits
        self.lambda_ = lambda_
        
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
        creator_fn = self.circuit_creators[self.kernel_type]
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
            qkernel = FidelityQuantumKernel(
                feature_map=isa_feature_map, 
                fidelity=fidelity, 
                max_circuits_per_job=300
            )
        else:
            transpiled_feature_map = transpile(self._feature_map, AerSimulator())
            qkernel = FidelityStatevectorKernel(feature_map=transpiled_feature_map)

        return qkernel