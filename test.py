import numpy as np 

v1 = np.random.rand(30)
v2 = np.random.rand(30)

print("_____________________")
print(v1)
print("_____________________")
print(v2)

dot = np.dot(v1, v2)
print("_____________________")
print(dot)
sume = 0

for i in range(0, 30):
    sume += v1[i] * v2[i]

print("_____________________")
print(sume)