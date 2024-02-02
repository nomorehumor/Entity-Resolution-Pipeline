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


def deduplicate_datasets(df_acm_deduplicated, df_dblp_deduplicated, clusters):
    idx_acm, idx_dblp = [], []
    entities_to_preserve = []

    for key in clusters.keys():
        if len(clusters[key]) > 2:
            id_acm = [el[2:] for el in clusters[key] if el.startswith('1')]
            idx_acm.extend(id_acm[1:])
            entities_to_preserve.append(id_acm[0])
            
            id_dblp = [el[2:] for el in clusters[key] if el.startswith('2')]
            idx_dblp.extend(id_dblp)
    
    if any("Combined" in col for col in df_acm_deduplicated.columns):
        df_acm_deduplicated = df_acm_deduplicated.drop(columns=[col for col in df_acm_deduplicated.columns if "Combined" in col])
    if any("Combined" in col for col in df_dblp_deduplicated.columns):
        df_dblp_deduplicated = df_dblp_deduplicated.drop(columns=[col for col in df_dblp_deduplicated.columns if "Combined" in col])
        
    df_acm_deduplicated = df_acm_deduplicated[~df_acm_deduplicated['paperId_acm'].isin(idx_acm)]
    df_dblp_deduplicated = df_dblp_deduplicated[~df_dblp_deduplicated['paperId_dblp'].isin(idx_dblp)]
    
    df_entities_to_preserve = df_acm_deduplicated[df_acm_deduplicated['paperId_acm'].isin(entities_to_preserve)]
    df_entities_to_preserve.rename(columns={
                                    'paperId_acm': 'paperId_dblp', 
                                    'title_acm': 'title_dblp', 
                                    'authors_acm': 'authors_dblp', 
                                    'venue_acm': 'venue_dblp', 
                                    'year_acm': 'year_dblp'
                                    }, inplace=True)
    df_dblp_deduplicated = pd.concat([df_dblp_deduplicated, df_entities_to_preserve], axis=1)
    
    return df_acm_deduplicated, df_dblp_deduplicated

