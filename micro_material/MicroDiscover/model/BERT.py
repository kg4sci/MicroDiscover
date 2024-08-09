import torch
from MicroDiscover.graph_process.process_graph import graph_from_neo4j
import numpy as np
from transformers import BertTokenizer, BertModel


if __name__ == "__main__":
    knowledge_graph = graph_from_neo4j()
    node_ids = [node[0] for node in knowledge_graph.nodes(data=True)]
    id_mapping = {node_id: i for i, node_id in enumerate(node_ids)}

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

    loaded_node_vectors_array = np.load('bert_node_vectors.npy')
    print(loaded_node_vectors_array.shape)
    microbe_names = []
    microbe_vector = []
    for node in knowledge_graph.nodes(data=True):
        if node[1]["properties"]["defineName"] == "ncbi_species":
            new_id = id_mapping[node[0]]
            microbe_names.append(node[1]["properties"]["name"])
            microbe_vector.append(loaded_node_vectors_array[new_id])

    dtype = [('name', 'U50'), ('vector', float, loaded_node_vectors_array.shape[1])]
    structured_array = np.zeros(len(microbe_names), dtype=dtype)
    structured_array['name'] = microbe_names
    structured_array['vector'] = microbe_vector
    np.save('BERT_ncbi_vectors.npy', structured_array)



