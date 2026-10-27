import numpy as np

def edge2mat(edges, num_node):
    matrix = np.zeros((num_node, num_node), dtype=np.float32)
    for source, target in edges:
        matrix[target, source] = 1.0
    return matrix

def normalize_digraph(matrix):
    degree = matrix.sum(axis=0, keepdims=True)
    return np.divide(matrix, degree, out=np.zeros_like(matrix), where=degree > 0)

def get_spatial_graph(num_node, self_link, inward, outward):
    return np.stack([
        edge2mat(self_link, num_node),
        normalize_digraph(edge2mat(inward, num_node)),
        normalize_digraph(edge2mat(outward, num_node)),
    ]).astype(np.float32)
