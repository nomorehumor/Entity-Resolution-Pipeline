def create_undirected_bipartite_graph(matched_pairs):
    graph = {}
    for idx1, idx2 in zip(matched_pairs['idx1'], matched_pairs['idx2']):
        node1 = f"1_{idx1}"
        node2 = f"2_{idx2}"
        if node1 not in graph:
            graph[node1] = set()
        if node2 not in graph:
            graph[node2] = set()
        graph[node1].add(node2)
        graph[node2].add(node1)
    return graph

def connected_components(matched_pairs):

    graph = create_undirected_bipartite_graph(matched_pairs)
    
    def dfs(graph, start_node, traversed_nodes, current_component):
        traversed_nodes.add(start_node)
        current_component.add(start_node)
        for node in graph[start_node]:
            if node not in traversed_nodes:
                dfs(graph, node, traversed_nodes, current_component)

    connected_components = {}
    traversed_nodes = set()
    for node in graph.keys():
        if node not in traversed_nodes:
            current_component = set([node])
            dfs(graph, node, traversed_nodes, current_component)
            connected_components[node] = current_component
            
    return connected_components
