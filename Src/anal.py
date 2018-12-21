## Data Analysis Code for Parameter Optimization for Deep Neural Network
## School of Software, Hallym University
##
## Jeong-Gun Lee (jeonggun.lee@gmail.com)
##
import numpy as np
#arr = np.loadtxt("all_p_ori.txt", delimiter=' ')
#arr = np.loadtxt("no_opt.txt", delimiter=' ')
#arr = np.loadtxt("all_p_bit_optimization.txt", delimiter=' ')
arr = np.loadtxt("all_p_opt_zero_elim.txt", delimiter=' ')
#arr = np.loadtxt("all_p.txt", delimiter=' ')
dims = len(arr.shape)
size = 1

for i in range(dims):
    size = arr.shape[i]*size

b = arr.reshape(size)
unique, counts = np.unique(b, return_counts=True)
print(len(unique))
print(unique)
print(counts)

for i in range(len(unique)):
    print(counts[i], unique[i])

print("# of P :", len(unique))


positive = unique[ unique > 0 ]
negative = unique[ unique < 0 ]

print("# of positive P :", len(positive))
print("# of negative P :", len(negative))

negative = -1*negative

pos_list = positive.tolist()
neg_list = negative.tolist()

set_inter = set(pos_list) &  set(neg_list)
inter = list(set_inter)

print("# of intersection: :", len(inter))
