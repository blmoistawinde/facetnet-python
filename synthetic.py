#coding=utf-8
import networkx as nx
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def comm_exchanging(comm_affli,num_changing=3):
    idx = np.random.choice(np.arange(len(comm_affli)),num_changing,replace=False)
    for node in idx:
        comm_affli[node] = (comm_affli[node] + np.random.randint(1,4)) % 4
    return comm_affli

def edge_aligning(comm_affli,z,avg_degree=16,idmap=[]):
    from itertools import combinations
    if idmap == []: idmap = [i for i in range(len(comm_affli))]
    G = nx.Graph()
    within_edges, inter_edges = [],[]
    for u,v in combinations(range(len(comm_affli)),2):
        if comm_affli[u] == comm_affli[v]:
            within_edges.append((u,v))
        else:
            inter_edges.append((u, v))
    within_edges_id = np.random.choice(range(len(within_edges)),int((avg_degree-z)*len(comm_affli)/2),replace=False)
    inter_edges_id = np.random.choice(range(len(inter_edges)), int(z*len(comm_affli)/2), replace=False)
    for i in within_edges_id:
        u,v = within_edges[i]
        G.add_edge(idmap[u],idmap[v])
    for i in inter_edges_id:
        u,v = inter_edges[i]
        G.add_edge(idmap[u],idmap[v])
    return G

def edge_aligning_weighted(comm_affli,z,avg_degree=16,idmap=[]):
    from itertools import combinations
    if idmap == []: idmap = [i for i in range(len(comm_affli))]
    G = nx.Graph()
    within_edges, inter_edges = [],[]
    for u,v in combinations(range(len(comm_affli)),2):
        if comm_affli[u] == comm_affli[v]:
            within_edges.append((u,v))
        else:
            inter_edges.append((u, v))
    within_edges_id = np.random.choice(range(len(within_edges)),int((avg_degree-z)*len(comm_affli)/2),replace=False)
    inter_edges_id = np.random.choice(range(len(inter_edges)), int(z*len(comm_affli)/2), replace=False)
    for i in within_edges_id:
        u,v = within_edges[i]
        G.add_edge(idmap[u],idmap[v],weight=2)
    for i in inter_edges_id:
        u,v = inter_edges[i]
        G.add_edge(idmap[u],idmap[v],weight=1)
    return G

def plot_graph_with_comm(G,comm_affli):
    colors = ["r","g","b","purple"]
    pos = nx.spring_layout(G)
    for comm in range(np.max(comm_affli)):
        nx.draw_networkx_nodes(G, pos, nodelist=np.where(comm_affli==comm)[0].tolist(),
                               node_color=colors[comm])
    nx.draw_networkx_edges(G, pos)
    plt.show()

def comm_exchanging2(idmax,comm_affli,idmap,idmap_inv):
    try:
        idx = np.random.choice(np.where(comm_affli == 3)[0],4,replace=False)
        comm_affli = np.array([comm for i,comm in enumerate(comm_affli) if not i in idx])
        idmap = [nid for i,nid in enumerate(idmap) if not i in idx]
        idmap_inv = {nid: i for i, nid in enumerate(idmap)}
    except:
        pass
    idmap += [idmax, idmax + 1]
    idmap_inv[idmax] = len(comm_affli)
    idmap_inv[idmax+1] = len(comm_affli)+1
    comm_affli = np.array(comm_affli.tolist()+[0,0])
    idmax += 2
    return idmax,comm_affli,idmap,idmap_inv

# generate an evolutional graph as is stated in 4.1.2
def generate_evolution(output_path,z=3,avg_degree=16,tsteps=10):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    comm_affli = (np.arange(0, 128) / 32).astype(int)
    np.random.shuffle(comm_affli)
    for t in range(tsteps):
        G = edge_aligning(comm_affli, z, avg_degree)
        nx.write_weighted_edgelist(G, output_path+'%d.edgelist' % t)
        with open(output_path+"%d.comm" % t,"w") as f:
            f.write("\n".join("%d %d" % (i, comm) for i,comm in enumerate(comm_affli)))
        comm_affli = comm_exchanging(comm_affli,num_changing=3)
        # uncomment this line to plot the snapshot graph
        # plot_graph_with_comm(G, comm_affli)
    pass

# generate an evolutional graph such that at each timestamp:
# 1. 4 nodes of community 3 was removed
# 2. 2 new nodes of community 0 was added (with increased node id)
def generate_evolution2(output_path,z=3,avg_degree=16,tsteps=9):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    comm_affli = (np.arange(0, 128) / 32).astype(int)
    np.random.shuffle(comm_affli)
    idmap = [i for i in range(128)]                     # 数组下标与结点唯一id标识的映射
    idmap_inv = {nid: i for i,nid in enumerate(idmap)}  # 结点唯一id标识与数组下标的映射
    idmax = 128                                      # 当前最大编号+1【出现过的总人次】
    for t in range(tsteps):
        G = edge_aligning(comm_affli ,z, avg_degree, idmap)
        nx.write_weighted_edgelist(G, output_path+'%d.edgelist' % t)
        with open(output_path+"%d.comm" % t,"w") as f:
            f.write("\n".join("%d %d" % (idmap[i], comm) for i,comm in enumerate(comm_affli)))
        idmax,comm_affli,idmap,idmap_inv = comm_exchanging2(idmax,comm_affli,idmap,idmap_inv)
    pass


# generate an evolutional graph as is stated in 4.1.2, adding weight
def generate_evolution3(output_path, z=3, avg_degree=16, tsteps=10):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    comm_affli = (np.arange(0, 128) / 32).astype(int)
    np.random.shuffle(comm_affli)
    for t in range(tsteps):
        G = edge_aligning_weighted(comm_affli, z, avg_degree)
        nx.write_weighted_edgelist(G, output_path + '%d.edgelist' % t)
        with open(output_path + "%d.comm" % t, "w") as f:
            f.write("\n".join("%d %d" % (i, comm) for i, comm in enumerate(comm_affli)))
        comm_affli = comm_exchanging(comm_affli, num_changing=3)
        # uncomment this line to plot the snapshot graph
        # plot_graph_with_comm(G, comm_affli)
    pass


if __name__ == "__main__":
    generate_evolution("./data/syntetic1/", tsteps=10)
    generate_evolution2("./data/syntetic2/", tsteps=9)
