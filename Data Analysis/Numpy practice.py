import numpy as np

# matrix : 
# [[ 2  4  6  8 10]
#  [ 3  6  9 12 15]
#  [ 5 10 15 20 25]
#  [ 7 14 21 28 35]
#  [11 22 33 44 55]]

def solution():
    prime = [2, 3, 5, 7, 11]

    prime_np = np.array(prime*5).reshape(5,5).transpose()
    # [[ 2  2  2  2  2]
    # [ 3  3  3  3  3]
    # [ 5  5  5  5  5]
    # [ 7  7  7  7  7]
    # [11 11 11 11 11]]
    mul = [1,2,3,4,5]
    mul_np = np.array(mul)
    matrix = prime_np * mul_np

    matrix_dia = np.diagonal(matrix)
    dia_sum = np.sum(matrix_dia)
    dia_mean = np.mean(matrix_dia)

    return matrix, dia_sum, dia_mean

def print_answer(**kwargs):
    for key in kwargs.keys():
        print(key,":", kwargs[key])

matrix, dia_sum, dia_mean = solution()

print_answer(matrix=matrix, dia_sum=dia_sum, dia_mean=dia_mean)
