import itertools
from qiskit import QuantumCircuit
class QuantumKernelCircuits:

    def __init__(self, n_qubits, lambda_=1.0):
        self.lambda_ = lambda_
        self.n_qubits = n_qubits
    
    @staticmethod
    def create_iqp_full(self, params):
        # params = ParameterVector("x", n_qubits)
        qc = QuantumCircuit(self.n_qubits)
        
        # Hadamard layer
        for i in range(self.n_qubits):
            qc.h(i)
            
        # Single-qubit rotation layer
        for i in range(self.n_qubits):
            qc.rz(params[i], i)
            
        # Full entanglement layer
        for i, j in itertools.combinations(range(self.n_qubits), 2):
            theta = self.lambda_ * params[i] * params[j]
            qc.rzz(theta, i, j)
                
        return qc
    
    @staticmethod
    def create_iqp_linear(self, params):
        qc = QuantumCircuit(self.n_qubits)
        
        # Hadamard layer
        for i in range(self.n_qubits):
            qc.h(i)
            
        # Single-qubit rotation layer
        for i in range(self.n_qubits):
            qc.rz(params[i], i)
            
        # Linear entanglement layer
        for i in range(self.n_qubits - 1):
            theta = self.lambda_ * params[i] * params[i+1]
            qc.rzz(theta, i, i+1)
        #     qc.cx(i, i+1)
        #     qc.rz(x[i] * x[i+1], i+1)
        #     qc.cx(i, i+1)
                
        return qc
    
    @staticmethod
    def create_iqp_circular(self, params):
        qc = QuantumCircuit(self.n_qubits)
        
        # Hadamard layer
        for i in range(self.n_qubits):
            qc.h(i)
            
        # Single-qubit rotation layer
        for i in range(self.n_qubits):
            qc.rz(params[i], i)
            
        # Circular entanglement layer
        for i in range(self.n_qubits):
            j = (i + 1) % self.n_qubits
            theta = self.lambda_ * params[i] * params[j]
            qc.rzz(theta, i, j)
                
        return qc
    
    @staticmethod
    def create_pauli_x(self, params):
        qc = QuantumCircuit(self.n_qubits)
        
        for i in range(self.n_qubits):
            qc.rx(params[i], i)
            
        return qc
    
    @staticmethod
    def create_pauli_y(self, params):
        qc = QuantumCircuit(self.n_qubits)
        
        for i in range(self.n_qubits):
            qc.ry(params[i], i)
            
        return qc
    
    @staticmethod
    def create_pauli_z(self, params):
        qc = QuantumCircuit(self.n_qubits)
        
        for i in range(self.n_qubits):
            qc.h(i)
            qc.rz(params[i], i)
            
        return qc