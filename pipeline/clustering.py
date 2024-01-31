def create_undirected_bipartite_graph(matched_pairs):
    graph = {}
    for idx1, idx2 in zip(matched_pairs['index_acm'], matched_pairs['index_dblp']):
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


def deduplicate_datasets(df_acm, df_dblp, clusters):
    idx_acm, idx_dblp = [], []

    for key in clusters.keys():
        if len(clusters[key]) > 2:
            id_acm = [int(el[2:]) for el in clusters[key] if el.startswith('1')]
            idx_acm.extend(id_acm[1:])
            id_dblp = [int(el[2:]) for el in clusters[key] if el.startswith('2')]
            idx_dblp.extend(id_dblp[1:])
    
    if any("Combined" in col for col in df_acm.columns):
        df_acm = df_acm.drop(columns=[col for col in df_acm.columns if "Combined" in col])
    if any("Combined" in col for col in df_dblp.columns):
        df_dblp = df_dblp.drop(columns=[col for col in df_dblp.columns if "Combined" in col])
    
    return df_acm.drop(idx_acm), df_dblp.drop(idx_dblp)

