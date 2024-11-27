import numpy as np
from scipy.linalg import solve_continuous_lyapunov, svd, sqrtm, inv

def balanced_realization(A, B):
    """
    Computes the balanced realization of a system given its A and B matrices.
    
    Parameters:
    - A: np.ndarray, system matrix (n x n)
    - B: np.ndarray, input matrix (n x m)
    
    Returns:
    - A_hat: np.ndarray, balanced realization of A
    - B_hat: np.ndarray, balanced realization of B
    - info: dict, containing gramian matrices (P, Q) and the similarity transform (T)
    """
    # Compute the controllability Gramian P
    P = solve_continuous_lyapunov(A, -B @ B.T)
    
    # Compute the observability Gramian Q
    Q = solve_continuous_lyapunov(A.T, -np.eye(A.shape[0]))
    
    # Singular value decomposition of P @ Q
    U, Sigma, Vh = svd(P @ Q)
    
    # Compute the square roots of the singular values (Hankel singular values)
    Sigma_sqrt_inv = np.diag(1.0 / np.sqrt(Sigma))
    Sigma_sqrt = np.diag(np.sqrt(Sigma))
    
    # Compute the similarity transform T
    T = sqrtm(P) @ U @ Sigma_sqrt_inv
    
    # Compute the inverse of the similarity transform
    T_inv = inv(T)
    
    # Transform A and B to their balanced form
    A_hat = T_inv @ A @ T
    B_hat = T_inv @ B
    
    # Store additional info in a dictionary
    info = {
        "P": P,
        "Q": Q,
        "T": T,
        "Hankel_singular_values": Sigma
    }
    
    return A_hat, B_hat, info