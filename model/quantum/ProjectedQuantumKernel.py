"""
Projected Quantum Kernel (PQK) — implementation faithful to Huang et al. (2021).

Reference:
    Huang, H.-Y., Broughton, M., Mohseni, M., Babbush, R., Boixo, S., Neven, H.,
    & McClean, J. R. (2021). Power of data in quantum machine learning.
    Nature Communications, 12, 2631. https://doi.org/10.1038/s41467-021-22539-9

Definition (Eq. 9 of the paper):
    k_PQ(x_i, x_j) = exp(-gamma * sum_k ||rho_k(x_i) - rho_k(x_j)||_F^2)

where rho_k(x) = Tr_{j != k}[rho(x)] is the 1-particle reduced density matrix
(1-RDM) of qubit k for the state |phi(x)> = U_enc(x) |0>^n.

Implementation:
    Uses the Pauli basis decomposition of 1-qubit RDM:
        rho_k = (1/2)(I + <X_k>X + <Y_k>Y + <Z_k>Z)
    Then:
        ||rho_k(x) - rho_k(x')||_F^2 = (1/2) sum_{P in {X,Y,Z}} (<P_k>_x - <P_k>_x')^2
    So PQK becomes an RBF kernel in R^(3n) feature space of Pauli expectation values.

Complexity:
    - Quantum: O(N) statevector evaluations (vs O(N^2) for fidelity-based kernels)
    - Classical: O(N^2 * 3n) for RBF Gram matrix computation
"""

import numpy as np
from qiskit.quantum_info import Statevector, SparsePauliOp
from sklearn.metrics.pairwise import rbf_kernel


class ProjectedQuantumKernel:
    """
    Projected Quantum Kernel implementing Huang et al. 2021 Eq. 9.
    
    The API mirrors FidelityStatevectorKernel from qiskit-machine-learning,
    so it can be used as a drop-in replacement in code expecting `.evaluate(X, Y)`.
    
    Parameters
    ----------
    feature_map : QuantumCircuit
        Parameterized quantum circuit U_enc(x) that encodes classical x into |phi(x)>.
    gamma : float, default=1.0
        RBF bandwidth in the projected feature space. Higher gamma = narrower kernel.
    cache : bool, default=True
        Cache projected features by id(X) to avoid recomputation across calls.
    
    Examples
    --------
    >>> from qiskit.circuit import ParameterVector
    >>> from qiskit.circuit.library import ZZFeatureMap
    >>> fm = ZZFeatureMap(feature_dimension=4, reps=2)
    >>> pqk = ProjectedQuantumKernel(feature_map=fm, gamma=1.0)
    >>> K_train = pqk.evaluate(X_train)             # symmetric Gram matrix
    >>> K_test = pqk.evaluate(X_test, X_train)      # rectangular for inference
    """
    
    def __init__(self, feature_map, gamma=1.0, cache=True):
        self.feature_map = feature_map
        self.gamma = float(gamma)
        self.cache = cache
        self.n_qubits = feature_map.num_qubits
        
        # Pre-build all 3n Pauli observables (X_k, Y_k, Z_k for each qubit k)
        # This avoids re-building observables every time we project.
        self._observables = self._build_pauli_observables()
        
        # Cache for projected features (keyed by id of input array)
        self._cache = {}
    
    def _build_pauli_observables(self):
        """Build [X_0, Y_0, Z_0, X_1, Y_1, Z_1, ...] as SparsePauliOp."""
        observables = []
        for q in range(self.n_qubits):
            for pauli in ['X', 'Y', 'Z']:
                op = SparsePauliOp.from_sparse_list(
                    [(pauli, [q], 1.0)], num_qubits=self.n_qubits
                )
                observables.append(op)
        return observables
    
    def _project(self, X):
        """
        Project each x in X to the 3n-dim feature vector of Pauli expectations.
        
        Returns
        -------
        Phi : ndarray of shape (N, 3 * n_qubits)
            Phi[i] = [<X_0>_xi, <Y_0>_xi, <Z_0>_xi, <X_1>_xi, ...]
        """
        # Check cache first to avoid recomputation
        if self.cache:
            key = id(X)
            if key in self._cache:
                return self._cache[key]
        
        X = np.asarray(X)
        n_samples = X.shape[0]
        n_features_out = 3 * self.n_qubits
        Phi = np.zeros((n_samples, n_features_out), dtype=np.float64)
        
        for i, x in enumerate(X):
            # Bind parameters and compute the full statevector |phi(x)>
            bound = self.feature_map.assign_parameters(x)
            sv = Statevector(bound)
            
            # Compute <P_k>_x for each (k, P) — this is equivalent to
            # tracing out other qubits then measuring P on qubit k.
            for j, obs in enumerate(self._observables):
                Phi[i, j] = sv.expectation_value(obs).real
        
        if self.cache:
            self._cache[id(X)] = Phi
        
        return Phi
    
    def evaluate(self, x_vec, y_vec=None):
        """
        Compute PQK Gram matrix K[i, j] = k_PQ(x_i, y_j).
        
        Parameters
        ----------
        x_vec : ndarray of shape (N_x, n_features)
        y_vec : ndarray of shape (N_y, n_features) or None
            If None, computes symmetric K(X, X).
        
        Returns
        -------
        K : ndarray of shape (N_x, N_y) (or (N_x, N_x) if y_vec is None)
        """
        Phi_x = self._project(x_vec)
        Phi_y = Phi_x if y_vec is None else self._project(y_vec)
        # RBF kernel: exp(-gamma * ||phi_x - phi_y||^2)
        return rbf_kernel(Phi_x, Phi_y, gamma=self.gamma)
    
    def clear_cache(self):
        """Clear the feature cache (call between independent experiments)."""
        self._cache = {}