import itertools
import numpy as np
from qiskit import QuantumCircuit

class CustomGateCircuits:
    def __init__(self, n_qubits, lambda_=1.0):
        self.lambda_ = lambda_
        self.n_qubits = n_qubits

    # X + RXX
    def x_full(self, params):
        qc = QuantumCircuit(self.n_qubits)
        
        # Hadamard layer
        for i in range(self.n_qubits):
            qc.h(i)
            
        # Single-qubit rotation layer
        for i in range(self.n_qubits):
            qc.rx(params[i], i)
            
        # Full entanglement layer
        for i, j in itertools.combinations(range(self.n_qubits), 2):
            theta = self.lambda_ * params[i] * params[j]
            qc.rxx(theta, i, j)
                
        return qc
    
    def x_linear(self, params):
        qc = QuantumCircuit(self.n_qubits)
        
        # Hadamard layer
        for i in range(self.n_qubits):
            qc.h(i)
            
        # Single-qubit rotation layer
        for i in range(self.n_qubits):
            qc.rx(params[i], i)
            
        # Linear entanglement layer
        for i in range(self.n_qubits - 1):
            theta = self.lambda_ * params[i] * params[i+1]
            qc.rxx(theta, i, i+1)
        
        return qc
    
    def x_circular(self, params):
        qc = QuantumCircuit(self.n_qubits)
        
        # Hadamard layer
        for i in range(self.n_qubits):
            qc.h(i)
            
        # Single-qubit rotation layer
        for i in range(self.n_qubits):
            qc.rx(params[i], i)
            
        # Circular entanglement layer
        for i in range(self.n_qubits):
            j = (i + 1) % self.n_qubits
            theta = self.lambda_ * params[i] * params[j]
            qc.rxx(theta, i, j)
        
        return qc
    
    # Y + RYY
    def y_full(self, params):
        qc = QuantumCircuit(self.n_qubits)
        
        # Hadamard layer
        for i in range(self.n_qubits):
            qc.h(i)
            
        # Single-qubit rotation layer
        for i in range(self.n_qubits):
            qc.ry(params[i], i)
            
        # Full entanglement layer
        for i, j in itertools.combinations(range(self.n_qubits), 2):
            theta = self.lambda_ * params[i] * params[j]
            qc.ryy(theta, i, j)
                
        return qc
    
    def y_linear(self, params):
        qc = QuantumCircuit(self.n_qubits)
        
        # Hadamard layer
        for i in range(self.n_qubits):
            qc.h(i)
            
        # Single-qubit rotation layer
        for i in range(self.n_qubits):
            qc.ry(params[i], i)
            
        # Linear entanglement layer
        for i in range(self.n_qubits - 1):
            theta = self.lambda_ * params[i] * params[i+1]
            qc.ryy(theta, i, i+1)
        
        return qc
    
    def y_circular(self, params):
        qc = QuantumCircuit(self.n_qubits)
        
        # Hadamard layer
        for i in range(self.n_qubits):
            qc.h(i)
            
        # Single-qubit rotation layer
        for i in range(self.n_qubits):
            qc.ry(params[i], i)
            
        # Circular entanglement layer
        for i in range(self.n_qubits):
            j = (i + 1) % self.n_qubits
            theta = self.lambda_ * params[i] * params[j]
            qc.ryy(theta, i, j)
        
        return qc