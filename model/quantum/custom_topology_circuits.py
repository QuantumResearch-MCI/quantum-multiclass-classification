import itertools
import numpy as np
from qiskit import QuantumCircuit

class CustomTopologyCircuits:
    def __init__(self, n_qubits, lambda_=1.0):
        self.lambda_ = lambda_
        self.n_qubits = n_qubits

    def star(self, params):
        qc = QuantumCircuit(self.n_qubits)

        for i in range(self.n_qubits):
            qc.h(i)
            qc.rz(params[i], i)

        # hub star = qubit 0
        hub = 0
        for j in range(self.n_qubits):
            if j == hub:
                continue
            alpha = self.lambda_ * params[hub]
            beta = self.lambda_ * params[j]
            gamma = self.lambda_ * params[hub] * params[j]

            qc.cx(hub, j)
            qc.rx(alpha, j)
            qc.ry(beta, j)
            qc.rz(gamma, j)
            qc.cx(hub, j)
        
        return qc