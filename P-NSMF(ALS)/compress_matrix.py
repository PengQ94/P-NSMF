import numpy as np


def compress_symmetric_matrix(matrix, dimension):
    return matrix[np.triu_indices(dimension)]


def recover_symmetric_matrix(matrix_triu, dimension):
    matrix = np.zeros([dimension, dimension])
    matrix[np.triu_indices(dimension)] = matrix_triu
    matrix = matrix + matrix.T - np.diag(matrix.diagonal())

    return matrix
