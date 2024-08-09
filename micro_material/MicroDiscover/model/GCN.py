import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import GCNConv
from process_graph import graph_from_neo4j
import networkx as nx
import numpy as np
from py2neo import Graph
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, BertModel


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        # self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.linear = nn.Linear(768, 768)
        self.relu = nn.ReLU()

    def forward(self, feature, edge_index):
        x = self.conv1(feature, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.linear(x)
        x = self.relu(x)
        concatenated_tensor = torch.cat((feature, x), dim=1)
        return concatenated_tensor


def compute_dot_product(microbe_vectors, material_vectors):

    if microbe_vectors.shape != material_vectors.shape:
        raise ValueError("error")

    dot_products = np.einsum('ij,ij->i', microbe_vectors, material_vectors)
    return dot_products

def normalize_scores(mrr_scores):
    min_score = min(mrr_scores)
    max_score = max(mrr_scores)

    normalized_scores = [(score - min_score) / (max_score - min_score) for score in mrr_scores]

    epsilon = 1e-10
    scaled_scores = [score * (1 - 2 * epsilon) + epsilon for score in normalized_scores]

    return scaled_scores


if __name__ == '__main__':
    knowledge_graph = graph_from_neo4j()
    nodes_a = [n for n, d in knowledge_graph.nodes(data=True) if d["labels"][0] == "material"]
    node_ids = [node[0] for node in knowledge_graph.nodes(data=True)]
    id_mapping = {node_id: i for i, node_id in enumerate(node_ids)}
    num_nodes = len(node_ids)
    new_adj_matrix = np.zeros((num_nodes, num_nodes))
    for edge in knowledge_graph.edges():
        src_id = id_mapping[edge[0]]
        dst_id = id_mapping[edge[1]]
        new_adj_matrix[src_id, dst_id] = 1
    adj_matrix_tensor = torch.tensor(new_adj_matrix)

    edge_indices = torch.nonzero(adj_matrix_tensor, as_tuple=True)
    edge_index = torch.stack(edge_indices)
    nx.set_node_attributes(knowledge_graph, id_mapping, 'id')

    id_list = []
    name_list = []

    for node_a in nodes_a:

        ego_subgraph = nx.ego_graph(knowledge_graph, node_a, radius=2)

        for node_b in ego_subgraph.nodes():
            if knowledge_graph.nodes[node_b]["labels"][0] == "microorganism":

                name_a = knowledge_graph.nodes[node_a]["properties"]["name"]
                name_b = knowledge_graph.nodes[node_b]["properties"]["name"]

                id_list.append((node_a, node_b))
                name_list.append((name_a, name_b))

    material_microorganism_dict = {}
    material_microorganism_name_dict = {}
    for item in id_list:
        key, value = item
        if id_mapping[key] in material_microorganism_dict:
            material_microorganism_dict[id_mapping[key]].append(id_mapping[value])
        else:
            material_microorganism_dict[id_mapping[key]] = [id_mapping[value]]

    node_features_array = np.loadtxt('bert_all_features.txt', delimiter=',')
    ones_array = np.ones_like(node_features_array)
    node_features_tensor = torch.tensor(ones_array, dtype=torch.float)

    candidate_entities = []
    material_list = []
    for node, data in knowledge_graph.nodes(data=True):
        if data["properties"]["defineName"] == 'microorganism':
            mapping_id = id_mapping[node]
            candidate_entities.append(mapping_id)
    for node, data in knowledge_graph.nodes(data=True):
        if data["properties"]["defineName"] == 'material':
            mapping_id = id_mapping[node]
            material_list.append(mapping_id)

    candidate_mapping = {entity: idx for idx, entity in enumerate(candidate_entities, start=0)}

    model = GCN(input_dim=768, hidden_dim=128, output_dim=768)

    epochs = 100
    lr = 0.00003
    optimizer = Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(1, epochs):
        optimizer.zero_grad()
        out = model(node_features_tensor, edge_index)
        MRR_list = []
        hits_20 = 0
        hits_10 = 0
        hits_5 = 0
        for material_item in material_list:
            material_feature = out[material_item].detach().numpy().reshape(1, -1)
            mrr_scores = []
            similarity_scores_dict = {}
            for microorganism_item in candidate_entities:
                microorganism_feature = out[microorganism_item].detach().numpy().reshape(1, -1)
                similarity_score = compute_dot_product(material_feature, microorganism_feature)
                mrr_scores.append(similarity_score.item())
            sorted_indices = np.argsort(mrr_scores)[::-1]
            if material_item in material_microorganism_dict:
                candidate_microorganism_list = material_microorganism_dict[material_item]
            else:
                continue
            min_rank = float('inf')
            for microorganism_item in candidate_microorganism_list:
                candidate_item = candidate_mapping[microorganism_item]
                rank = np.where(sorted_indices == candidate_item)[0][0] + 1
                min_rank = min(min_rank, rank)
            MRR_list.append(1 / min_rank)
            if min_rank <= 20:
                hits_20 += 1
            if min_rank <= 10:
                hits_10 += 1
            if min_rank <= 5:
                hits_5 += 1
        avg_mrr = sum(MRR_list) / len(MRR_list)
        mrr_tensor = -torch.tensor(avg_mrr, dtype=torch.float, requires_grad=True)
        print(avg_mrr)
        print("hits_5", hits_5 / len(material_list))
        print("hits_10", hits_10 / len(material_list))
        print("hits_20", hits_20 / len(material_list))

        optimizer.zero_grad()
        mrr_tensor.backward()
        optimizer.step()


