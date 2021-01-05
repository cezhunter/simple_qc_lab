import numpy as np
from fractions import Fraction
import random


def find_factors(kron_prod):
    n = len(kron_prod)
    x = np.where(kron_prod == 1)[0][0]
    f = []
    bt = ([1, 0], [0, 1])
    while n > 1:
        f.append(bt[(x % n) // (n // 2)])
        n = n // 2
    return np.array(f)

def find_factors_hadamard(prod):
    y = 1/np.sqrt(2)
    if np.array_equal(np.around(prod, 1), np.array([1/2, 1/2, -1/2, -1/2])):
        return np.array([[y, -y], [y, y]])
    elif np.array_equal(np.around(prod, 1), np.array([1/2, -1/2, -1/2, 1/2])):
        return np.array([[y, -y], [y, -y]])
    elif np.array_equal(np.around(prod, 1), np.array([1/2, 1/2, 1/2, 1/2])):
        return np.array([[y, y], [y, y]])
    elif np.array_equal(np.around(prod, 1), np.array([1/2, -1/2, 1/2, -1/2])):
        return np.array([[y, y], [y, -y]])

def apply_x_gate(v):
    x_gate = [[0, 1], [1, 0]]
    return np.matmul(x_gate, v)

def apply_h_gate(v):
    y = 1/np.sqrt(2)
    h_gate = [[y, y], [y, -y]]
    return np.matmul(h_gate, v)

def apply_c_not(v):
    c_not_gate = [[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]]
    return np.matmul(c_not_gate, v)

def bb(v0, v1, r):
    if r == 0:
        return np.array([v1, v0])
    if r == 1:
        return np.array([v1, apply_x_gate(v0)])
    if r == 2:
        return find_factors_hadamard(apply_c_not(np.kron(v1,v0)))
    if r == 3:
        res = find_factors_hadamard(apply_c_not(np.kron(v1,v0)))
        v1 = res[0]
        v0 = apply_x_gate(res[1])
        return np.array([v1, v0])


v0 = [1, 0]
v0 = apply_x_gate(v0)
v0 = apply_h_gate(v0)

v1 = [0, 1]
v1 = apply_x_gate(v1)
v1 = apply_h_gate(v1)

v2 = [1, 0]
v2 = apply_x_gate(v2)
v2 = apply_h_gate(v2)

r = bb(v0,v1, 0)
print(apply_h_gate(r[0]), apply_h_gate(r[1]))
r = bb(v0,v1, 1)
print(apply_h_gate(r[0]), apply_h_gate(r[1]))
r = bb(v0,v1, 2)
print(apply_h_gate(r[0]), apply_h_gate(r[1]))
r = bb(v0,v1, 3)
print(apply_h_gate(r[0]), apply_h_gate(r[1]))

print("")

r = bb(v0,v2, 0)
print(apply_h_gate(r[0]), apply_h_gate(r[1]))
r = bb(v0,v2, 1)
print(apply_h_gate(r[0]), apply_h_gate(r[1]))
r = bb(v0,v2, 2)
print(apply_h_gate(r[0]), apply_h_gate(r[1]))
r = bb(v0,v2, 3)
print(apply_h_gate(r[0]), apply_h_gate(r[1]))