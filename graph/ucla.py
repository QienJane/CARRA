from .tools import get_spatial_graph

num_node = 20
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (2, 3), (4, 3), (5, 3), (6, 5), (7, 6), (8, 7), (9, 3), (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19)]
inward = [(i - 1, j - 1) for i, j in inward_ori_index]
outward = [(j, i) for i, j in inward]
neighbor = inward + outward

def get_adjacency():
    return get_spatial_graph(num_node, self_link, inward, outward)
