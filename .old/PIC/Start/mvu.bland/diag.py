
import numpy as np
import cvxpy as cvx



Matrix = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])

def is_diagonizable(matrix):

    eigenvalues = cvx.Variable(matrix.shape[0])
    eigenvectors = cvx.Variable(matrix.shape[0], matrix.shape[0])
    constraints = [(Matrix - eigenvalues @ np.eye(matrix.shape[0])) @ eigenvectors == 0]
    problem = cvx.Problem(cvx.Minimize(cvx.norm(eigenvalues, 1)), constraints)
    problem.solve()
    return problem.status, eigenvalues.value
    
eigvals, eigvecs = np.linalg.eig(Matrix)


print(eigvals)
# print(eigvecs)
