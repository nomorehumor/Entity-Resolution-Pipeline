import pandas as pd


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
    ids_to_preserve = []

    for key in clusters.keys():
        if len(clusters[key]) > 2:
            id_acm = [int(el[2:]) for el in clusters[key] if el.startswith('1')]
            idx_acm.extend(id_acm[1:])
            ids_to_preserve.append(id_acm[0])
            
            id_dblp = [int(el[2:]) for el in clusters[key] if el.startswith('2')]
            idx_dblp.extend(id_dblp)
    
        
    df_acm_deduplicated = df_acm[~df_acm['index_acm'].isin(idx_acm)]
    df_dblp_deduplicated = df_dblp[~df_dblp['index_dblp'].isin(idx_dblp)]
    
    df_entities_to_preserve = df_acm[df_acm['index_acm'].isin(ids_to_preserve)].rename(columns={
                                    'paperId_acm': 'paperId_dblp', 
                                    'title_acm': 'title_dblp', 
                                    'authors_acm': 'authors_dblp', 
                                    'venue_acm': 'venue_dblp', 
                                    'year_acm': 'year_dblp',
                                    'Combined_acm': 'Combined_dblp'
                                    })
    df_dblp_deduplicated = pd.concat([df_dblp_deduplicated, df_entities_to_preserve])
    df_dblp_deduplicated.drop(["index_acm"],axis=1, inplace=True)
    return df_acm_deduplicated, df_dblp_deduplicated

