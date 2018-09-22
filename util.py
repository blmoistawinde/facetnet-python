import numpy as np
import networkx as nx

def Sim(adj_mat, weighted,gamma=0.2):
    N = adj_mat.shape[0]
    sim = np.matrix(np.eye(N,N))
    if weighted:                           # as stated in Section 4.2
        sim = np.exp(-1.0/(gamma*adj_mat))
    else:
        # similarity measure for unweighted graph is not mentioned in the paper, below is my assumption, which seems to work.
        sim = adj_mat/2.0                   # if linked, sim=0.5
    for i in range(N):
        sim[i,i] = 1.0
    sim /= np.sum(sim)
    return sim

def KL(A,B):
    ret = 0.0
    for a in np.ravel(A):
        for b in np.ravel(B):
            ret += a*np.log(a/b) - a + b
    return ret

def soft_modularity(soft_comm,W):
    N = W.shape[0]
    ret = np.trace(soft_comm.T*W*soft_comm)
    one = np.matrix(np.ones((N,1)))
    ret -= np.array(one.T*W.T*soft_comm*soft_comm.T*W*one).squeeze()
    return ret

if __name__ == "__main__":
    print(KL(np.array([[1,0.5],[0.5,1]]),np.array([[1,0.2],[0.2,1]])))

