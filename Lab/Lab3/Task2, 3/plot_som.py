import numpy as np
import matplotlib.pyplot as plt

x = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,1,1,1,0,0,1,1,1,0,0,1,0,0,0,0,0,0,1,0,1,1,1,0,0,0,0,4,3,4,4,4,6,3,3,4,4,3,6,4,5,3,3,7,4,5,4,7,3,5,5,3,3,4,5,6,3,4,3,4,7,7,3,4,4,6,4,5,6,4,3,5,6,6,3,2,5,5,7,7,7,5,7,7,7,7,6,4,7,5,7,7,4,6,7,7,6,5,7,7,6,5,7,6,7,7,7,7,7,7,5,7,7,4,6,7,5,5,4,7,5,5,4,6,5,4,7]
y = [4,2,1,1,4,7,2,4,0,2,6,3,1,0,7,7,7,4,7,5,5,5,3,4,3,2,4,5,4,2,2,5,7,7,2,3,6,4,0,4,4,0,0,4,6,1,6,1,6,3,5,5,5,0,4,1,6,0,4,0,0,1,2,3,2,5,0,2,3,1,2,3,4,3,4,5,5,5,2,2,0,1,2,3,0,6,5,3,0,1,1,2,2,0,1,0,0,4,0,1,7,3,6,5,7,7,0,7,5,7,6,4,6,3,3,7,5,7,7,3,7,2,7,4,7,6,3,3,5,6,7,7,5,4,4,7,7,5,2,6,7,6,3,7,7,6,4,5,7,3]
truth = [0]*50 + [1]*50 + [2]*50 

x = np.array(x)
y = np.array(y)

plt.scatter(x, y, c=truth, cmap='viridis')
plt.show()


def count_appearance(x,y):
    # count the appearance of each pair of x,y
    count = {}
    for i in range(len(x)):
        if (x[i],y[i]) in count:
            count[(x[i],y[i])] += 1
        else:
            count[(x[i],y[i])] = 1

    # sort the count (high to low)
    count = dict(sorted(count.items(), key=lambda item: item[1], reverse=True))
    return count


print(count_appearance(x,y))
## (0,4), (7,7), (7,3)

from scipy.special import comb

def rand_index(actual, pred):
    tp_plus_fp = comb(np.bincount(actual), 2).sum()
    tp_plus_fn = comb(np.bincount(pred), 2).sum()
    A = np.c_[(actual, pred)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
            for i in set(actual))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)

