import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import RGCNConv
from process_graph import graph_from_neo4j
import networkx as nx
import numpy as np
from py2neo import Graph
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, BertModel


class RGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_relations):
        super(RGCN, self).__init__()
        self.conv1 = RGCNConv(input_dim, hidden_dim, num_relations)
        self.conv2 = RGCNConv(hidden_dim, output_dim, num_relations)
        self.linear = nn.Linear(768, 768)
        self.relu = nn.ReLU()

    def forward(self, feature, edge_index, edge_type):
        x = self.conv1(feature, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        # x = self.linear(x)
        concatenated_tensor = torch.cat((feature, x), dim=1)
        return concatenated_tensor

def compute_dot_product(microbe_vectors, material_vectors):

    if microbe_vectors.shape != material_vectors.shape:
        raise ValueError("微生物和材料节点向量的形状必须相同")

    dot_products = np.einsum('ij,ij->i', microbe_vectors, material_vectors)
    return dot_products


def normalize_scores(mrr_scores):
    min_score = min(mrr_scores)
    max_score = max(mrr_scores)

    normalized_scores = [(score - min_score) / (max_score - min_score) for score in mrr_scores]

    epsilon = 1e-10
    scaled_scores = [score * (1 - 2 * epsilon) + epsilon for score in normalized_scores]

    return scaled_scores


if __name__ == "__main__":
    knowledge_graph = graph_from_neo4j()
    nodes_a = [n for n, d in knowledge_graph.nodes(data=True) if d["labels"][0] == "material"]
    node_ids = [node[0] for node in knowledge_graph.nodes(data=True)]
    id_mapping = {node_id: i for i, node_id in enumerate(node_ids)}
    num_nodes = len(node_ids)
    new_adj_matrix = np.zeros((num_nodes, num_nodes))
    for edge in knowledge_graph.edges(data=True):
        src_id = id_mapping[edge[0]]
        dst_id = id_mapping[edge[1]]
        new_adj_matrix[src_id, dst_id] = 1
    adj_matrix_tensor = torch.tensor(new_adj_matrix)

    edge_index_list = []
    edge_type = []
    for edge in knowledge_graph.edges(data=True):
        src_id = id_mapping[edge[0]]
        dst_id = id_mapping[edge[1]]
        if adj_matrix_tensor[src_id, dst_id] == 1:
            edge_index_list.append([src_id, dst_id])
            edge_type.append(edge[2]['type'])
    edge_index_tensor = torch.tensor(edge_index_list).t().contiguous()
    edge_type_tensor = torch.tensor(edge_type, dtype=torch.long)
    print(edge_index_tensor)
    print("Edge types tensor:", edge_type_tensor)

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

    print(material_microorganism_dict)

    model_name = "bert"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    node_features = np.zeros((len(knowledge_graph.nodes), 768))
    for node in knowledge_graph.nodes(data=True):
        if "defineName" not in node[1]["properties"]:
            continue
        if node[1]["properties"]["defineName"] == "microorganism":
            new_id = id_mapping[node[0]]
            node_name = node[1]["properties"]["name"]
            node_mechanism = node[1]["properties"]["mechanism"]
            for item in [node_mechanism]:
                for value in item:
                    if isinstance(value, str):
                        node_name += value
            tokens = tokenizer(node_name, return_tensors='pt', truncation=True, max_length=12)
            with torch.no_grad():
                outputs = model(**tokens)
            node_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            node_features[new_id] = node_embedding

        if node[1]["properties"]["defineName"] == "material":
            new_id = id_mapping[node[0]]
            node_name = node[1]["properties"]["name"]
            tokens = tokenizer(node_name, return_tensors='pt', truncation=True, max_length=20)
            with torch.no_grad():
                outputs = model(**tokens)
            node_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            node_features[new_id] = node_embedding

        if node[1]["properties"]["defineName"] == "synthesis_method":
            new_id = id_mapping[node[0]]
            node_name = node[1]["properties"]["name"]
            tokens = tokenizer(node_name, return_tensors='pt', truncation=True, max_length=256)
            with torch.no_grad():
                outputs = model(**tokens)
            node_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            node_features[new_id] = node_embedding

        if node[1]["properties"]["defineName"] == "lineage":
            new_id = id_mapping[node[0]]
            node_name = node[1]["properties"]["name"]
            # node_name = ""
            node_mechanism = node[1]["properties"]["mechanism"]
            for item in [node_mechanism]:
                for value in item:
                    if isinstance(value, str):
                        node_name += value
            tokens = tokenizer(node_name, return_tensors='pt', truncation=True, max_length=48)
            with torch.no_grad():
                outputs = model(**tokens)
            node_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            node_features[new_id] = node_embedding

        if node[1]["properties"]["defineName"] == "element":
            new_id = id_mapping[node[0]]
            node_name = node[1]["properties"]["name"]
            tokens = tokenizer(node_name, return_tensors='pt', truncation=True, max_length=48)
            with torch.no_grad():
                outputs = model(**tokens)
            node_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            node_features[new_id] = node_embedding

        if node[1]["properties"]["defineName"] == "ncbi_species":
            new_id = id_mapping[node[0]]
            node_name = node[1]["properties"]["name"]
            node_mechanism = node[1]["properties"]["mechanism"]
            for item in [node_mechanism]:
                for value in item:
                    if isinstance(value, str):
                        node_name += value
            tokens = tokenizer(node_name, return_tensors='pt', truncation=True, max_length=48)
            with torch.no_grad():
                outputs = model(**tokens)
            node_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            node_features[new_id] = node_embedding

    node_features_array = np.array(node_features)
    node_features_array = np.loadtxt('bert_all_features.txt', delimiter=',')
    node_features_tensor = torch.tensor(node_features_array, dtype=torch.float)
    np.save('bert_node_vectors.npy', node_features_tensor)
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
    model = RGCN(input_dim=768, hidden_dim=128, output_dim=768, num_relations=3)
    epochs = 100
    lr = 0.003
    optimizer = Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(1, epochs):
        optimizer.zero_grad()
        out = model(node_features_tensor, edge_index_tensor, edge_type_tensor)
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

