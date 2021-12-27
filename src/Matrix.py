import numpy as np

def multiply(matrix1,matrix2):
    result_matrix = np.zeros((matrix1.shape[0],matrix2.shape[1]))
    for i in range(result_matrix.shape[0]):
        for j in range(result_matrix.shape[1]):
            for element in range(matrix1.shape[1]):
                result_matrix[i][j] += matrix1[i][element] * matrix2[element][j]
    return result_matrix

def transpose(matrix):
    return np.array([[matrix[i][j] for i in range(matrix.shape[0])] for j in range(matrix.shape[1])])
