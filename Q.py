import torch
import networkx as nx

def compute_modularity(graph, labels):
    # Convert labels to a dictionary of communities
    communities = {}
    for node, label in enumerate(labels):
        if label.item() not in communities:
            communities[label.item()] = [node]
        else:
            communities[label.item()].append(node)

    # Convert the communities dictionary to a format compatible with NetworkX
    community_list = [nodes for _, nodes in communities.items()]

    # Calculate modularity
    modularity = nx.community.modularity(graph, community_list)
    return modularity

# Example usage
# G = nx.karate_club_graph()  # Example graph
# r_assign = torch.tensor([...])  # Example tensor containing community assignments
#
# modularity = calculate_modularity(G, r_assign)
# print("Modularity Q:", modularity)
