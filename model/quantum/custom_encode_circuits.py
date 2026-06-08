import itertools
import numpy as np
from qiskit import QuantumCircuit

class CustomEncodeCircuits:
    def __init__(self, n_qubits, lambda_=1.0):
        self.lambda_ = lambda_
        self.n_qubits = n_qubits

    
    # Custom circuit - quadratic encoding
    def _custom_phi_single(self, x):
        return x

    def _quadratic_phi_pair(self, xi, xj):
        return xi**2 * xj**2
    
    def create_iqp_full_quadratic(self, params):
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
            theta = self.lambda_ * (self._quadratic_phi_pair(params[i], params[j]))
            qc.rzz(theta, i, j)
                
        return qc
    
    # @staticmethod
    def create_iqp_linear_quadratic(self, params):
        qc = QuantumCircuit(self.n_qubits)
        
        # Hadamard layer
        for i in range(self.n_qubits):
            qc.h(i)
            
        # Single-qubit rotation layer
        for i in range(self.n_qubits):
            qc.rz(params[i], i)
            
        # Linear entanglement layer
        for i in range(self.n_qubits - 1):
            theta = self.lambda_ * (self._quadratic_phi_pair(params[i], params[i+1]))
            qc.rzz(theta, i, i+1)
        #     qc.cx(i, i+1)
        #     qc.rz(x[i] * x[i+1], i+1)
        #     qc.cx(i, i+1)
        

        return qc
    
    # @staticmethod
    def create_iqp_circular_quadratic(self, params):
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
            theta = self.lambda_ * (self._quadratic_phi_pair(params[i], params[j]))
            qc.rzz(theta, i, j)
        

        return qc
    
    # Custom circuit - cosine encoding
    def _cosine_phi_single(self, x):
        return np.cos(x)
    
    def _cosine_phi_pair(self, xi, xj):
        return np.cos(xi) * np.cos(xj)
    
    def create_iqp_full_cosine(self, params):
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
            theta = self.lambda_ * (self._cosine_phi_pair(params[i], params[j]))
            qc.rzz(theta, i, j)

        return qc
    
    def create_iqp_linear_cosine(self, params):
        qc = QuantumCircuit(self.n_qubits)

        # Hadamard layer
        for i in range(self.n_qubits):
            qc.h(i)

        # Single-qubit rotation layer
        for i in range(self.n_qubits):
            qc.rz(params[i], i)

        # Linear entanglement layer
        for i in range(self.n_qubits - 1):
            theta = self.lambda_ * (self._cosine_phi_pair(params[i], params[i+1]))
            qc.rzz(theta, i, i+1)

        return qc
    
    def create_iqp_circular_cosine(self, params):
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
            theta = self.lambda_ * (self._cosine_phi_pair(params[i], params[j]))
            qc.rzz(theta, i, j)

        return qc
    
    # Custom circuit - selisih encoding
    def _selisih_phi_single(self, x):
        return x
    
    def _selisih_phi_pair(self, xi, xj):
        return (xi - xj)
    
    def create_iqp_full_selisih(self, params):
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
            theta = self.lambda_ * (self._selisih_phi_pair(params[i], params[j]))
            qc.rzz(theta, i, j)

        return qc
    
    def create_iqp_linear_selisih(self, params):
        qc = QuantumCircuit(self.n_qubits)

        # Hadamard layer
        for i in range(self.n_qubits):
            qc.h(i)

        # Single-qubit rotation layer
        for i in range(self.n_qubits):
            qc.rz(params[i], i)

        # Linear entanglement layer
        for i in range(self.n_qubits - 1):
            theta = self.lambda_ * (self._selisih_phi_pair(params[i], params[i+1]))
            qc.rzz(theta, i, i+1)

        return qc
    
    def create_iqp_circular_selisih(self, params):
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
            theta = self.lambda_ * (self._selisih_phi_pair(params[i], params[j]))
            qc.rzz(theta, i, j)

        return qc

    # Polynomial (xi^3 * xj^3)
    def create_iqp_circular_polynomial(self, params):
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
            theta = self.lambda_ * (params[i]**3 * params[j]**3)
            qc.rzz(theta, i, j)

        return qc

    def create_iqp_linear_polynomial(self, params):
        qc = QuantumCircuit(self.n_qubits)

        # Hadamard layer
        for i in range(self.n_qubits):
            qc.h(i)

        # Single-qubit rotation layer
        for i in range(self.n_qubits):
            qc.rz(params[i], i)

        # Linear entanglement layer
        for i in range(self.n_qubits - 1):
            theta = self.lambda_ * (params[i]**3 * params[i+1]**3)
            qc.rzz(theta, i, i+1)

        return qc

    def create_iqp_full_polynomial(self, params):
        qc = QuantumCircuit(self.n_qubits)

        # Hadamard layer
        for i in range(self.n_qubits):
            qc.h(i)

        # Single-qubit rotation layer
        for i in range(self.n_qubits):
            qc.rz(params[i], i)

        # Full entanglement layer
        for i, j in itertools.combinations(range(self.n_qubits), 2):
            theta = self.lambda_ * (params[i]**3 * params[j]**3)
            qc.rzz(theta, i, j)

        return qc

    # Polynomial (xi^4 * xj^4)
    def create_iqp_circular_polynomial4(self, params):
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
            theta = self.lambda_ * (params[i]**4 * params[j]**4)
            qc.rzz(theta, i, j)

        return qc
    
    def create_iqp_linear_polynomial4(self, params):
        qc = QuantumCircuit(self.n_qubits)

        # Hadamard layer
        for i in range(self.n_qubits):
            qc.h(i)

        # Single-qubit rotation layer
        for i in range(self.n_qubits):
            qc.rz(params[i], i)

        # Linear entanglement layer
        for i in range(self.n_qubits - 1):
            theta = self.lambda_ * (params[i]**4 * params[i+1]**4)
            qc.rzz(theta, i, i+1)

        return qc
    
    def create_iqp_full_polynomial4(self, params):
        qc = QuantumCircuit(self.n_qubits)

        # Hadamard layer
        for i in range(self.n_qubits):
            qc.h(i)

        # Single-qubit rotation layer
        for i in range(self.n_qubits):
            qc.rz(params[i], i)

        # Full entanglement layer
        for i, j in itertools.combinations(range(self.n_qubits), 2):
            theta = self.lambda_ * (params[i]**4 * params[j]**4)
            qc.rzz(theta, i, j)

        return qc
        