from . import tools

def get_adjacency():
    num_nodes = 42
    self_links = [(i, i) for i in range(num_nodes)]
    inward = []
    for offset in (0, 21):
        for finger in (1, 5, 9, 13, 17):
            inward.append((offset + finger, offset))
            inward.extend((offset + joint, offset + joint - 1)
                          for joint in range(finger + 1, finger + 4))
    outward = [(j, i) for i, j in inward]
    return tools.get_spatial_graph(num_nodes, self_links, inward, outward)
