# -*- coding: utf-8 -*-
# @Time : 2023/10/09 16:17
# @Author : 刘宇峰
# @File : prediction_model
# @Software : PyCharm


import networkx as nx
import numpy as np
import pandas as pd
from neo4j import GraphDatabase
import torch
import random
import csv
import requests
from torch_geometric.utils import to_undirected
from sklearn.preprocessing import StandardScaler
from Bio import Entrez, SeqIO
from py2neo import Graph


def graph_from_neo4j():
    """
    Use networkx to obtain graph data from the Neo4j knowledge graph, including node data and edge data
    :return: Return the graph and data obtained from neo4j
    """
    uri = "bolt://10.0.87.16:7687"
    username = "neo4j"
    password = "12345678"

    driver = Graph(uri, auth=(username, password))
    nodes_query = """
        MATCH (n)
        RETURN id(n) AS id, labels(n) AS labels, properties(n) AS properties
        """
    edges_query = """
        MATCH ()-[r]->()
        RETURN id(startNode(r)) AS source, type(r) AS type, id(endNode(r)) AS target
        """

    knowledge_graph = nx.Graph()
    nodes = driver.run(nodes_query)
    for node in nodes:
        node_id = node["id"]
        if node["labels"] == "ncbi":
            continue
        node_labels = node["labels"]
        node_properties = node["properties"]
        knowledge_graph.add_node(node_id, labels=node_labels, properties=node_properties)

    edges = driver.run(edges_query)
    for edge in edges:
        if edge["type"] == "ncbi_species":
            continue
        source = edge["source"]
        target = edge["target"]
        edge_type = 3
        if edge["type"] == "contain":
            edge_type = 0
        elif edge["type"] == "new_lineage":
            edge_type = 1
        elif edge["type"] == "synthesis":
            edge_type = 2
        knowledge_graph.add_edge(source, target, type=edge_type)

    return knowledge_graph


def nodes_ids_mapping(nodes_ids):
    """
    对图谱中的节点id进行映射
    :return:映射的字典
    """
    id_mapping = {}
    new_id = 0
    for node_id in nodes_ids:
        id_mapping[node_id] = new_id
        new_id += 1
    return id_mapping


def lineage_microorganism_edges():
    """
    从微生物到发育树的路径
    :return:
    """
    uri = "bolt://10.0.87.16:7687"
    username = "neo4j"
    password = "12345678"

    driver = GraphDatabase.driver(uri, auth=(username, password))

    query = """
    MATCH (startNode:microorganism)-[*1..2]-(targetNode:lineage)
    WHERE targetNode.rank = "strain"
    RETURN startNode,targetNode
    """
    graph = nx.Graph()

    # 执行查询并将结果添加到图中
    edges = []
    with driver.session() as session:
        result = session.run(query)
        for record in result:
            source_node = record[0]["name"]
            target_node = record[1]["name"]
            edges.append((source_node, target_node))
    unique_edges = list(set(edges))
    csv_file = 'lineage_microorganism_edge.csv'
    # 将数据写入 CSV 文件
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['node1', 'node2'])  # 写入标题行
        writer.writerows(unique_edges)  # 写入数据行

    driver.close()


def graph_adjacency_matrix():
    """
    构造知识图谱的邻接矩阵
    :param knowledge_graph: 构建好的知识图谱
    :return: 知识图谱的邻接矩阵
    """
    knowledge_graph = graph_from_neo4j()
    nodes_ids = list(knowledge_graph.nodes())
    nodes_num = len(knowledge_graph.nodes)
    id_mapping = nodes_ids_mapping(nodes_ids)
    edges = knowledge_graph.edges
    adjacency_matrix = np.zeros((nodes_num, nodes_num))
    for edge in edges:
        node1, node2 = edge
        new_node1 = id_mapping.get(node1)
        new_node2 = id_mapping.get(node2)
        if new_node1 is not None and new_node2 is not None:
            adjacency_matrix[new_node1][new_node2] = 1
            adjacency_matrix[new_node2][new_node1] = 1
    return adjacency_matrix


def build_samples():
    """
    构造正负样本
    :return: 返回样本
    """
    adjacency_matrix = graph_adjacency_matrix()

    edge_index_pos = np.column_stack(np.argwhere(adjacency_matrix != 0))
    edge_index_pos = torch.tensor(edge_index_pos, dtype=torch.long)
    edge_index_neg = np.column_stack(np.argwhere(adjacency_matrix == 0))
    edge_index_neg = torch.tensor(edge_index_neg, dtype=torch.long)

    # 从正负样本中随机选取相同数量的边，形成平衡的训练集
    num_pos_edges_number = edge_index_pos.shape[1]
    selected_neg_edge_indices = torch.randint(high=edge_index_neg.shape[1], size=(num_pos_edges_number,),
                                              dtype=torch.long)
    edge_index_neg_selected = edge_index_neg[:, selected_neg_edge_indices]
    edg_index_all = torch.cat((edge_index_pos, edge_index_neg_selected), dim=1)
    y = torch.cat((torch.ones((edge_index_pos.shape[1], 1)),
                   torch.zeros((edge_index_neg_selected.shape[1], 1))), dim=0)  # 将所有y值设置为1,0
    # 将邻接矩阵转换为无向图（会将每条边复制成两条相反的边）
    edge_index = to_undirected(edge_index_pos)
    return edge_index_pos, edge_index_neg_selected, edge_index, y


def extract_data_from_neo4j():
    """
    从neo4j获取数据
    :return:
    """
    uri = "bolt://10.0.87.16:7687/browser/"
    username = "neo4j"
    password = "12345678"
    driver = GraphDatabase.driver(uri, auth=(username, password))
    with driver.session() as session:
        query = "MATCH (n1)-[:CONNECTED_TO]-(n2) RETURN id(n1), id(n2)"
        result = session.run(query)
        data = [(record["id(n1)"], record["id(n2)"]) for record in result]
    return data


def build_nodes_features():
    """
    构造节点特征
    :return: 返回构造好的节点特征
    """
    knowledge_graph = graph_from_neo4j()
    node_features = np.zeros((len(knowledge_graph.nodes), 9))
    for i, (_, node_data) in enumerate(knowledge_graph.nodes(data=True)):
        for j, attr_name in enumerate(["name", "defineName", "function", "mental_resistance", "habitat",
                                       "microbial_information", "location", "shape", "size"]):
            if attr_name in node_data:
                node_features[i, j] = 1.0
            else:
                node_features[i, j] = 0.0
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(node_features)
    one_dim_features = np.mean(scaled_features, axis=1, keepdims=True)
    nodes_features = torch.from_numpy(one_dim_features).float()
    return nodes_features


def split_dataset(y):
    """
    将数据拆分为训练集和测试集
    :param y: 数据集
    :return: 返回训练集和测试集
    """
    idx = np.arange(y.shape[0])
    np.random.shuffle(idx)
    train_idx = idx[:int(0.8 * len(idx))]
    test_idx = idx[int(0.2 * len(idx)):]
    return train_idx, test_idx


def cypher_neo4j_samples():
    """
    使用cypher语言去查询知识图谱构造新的正负样本
    :return:返回正负样本
    """
    uri = "bolt://10.0.87.16:7687"
    username = "neo4j"
    password = "12345678"
    driver = GraphDatabase.driver(uri, auth=(username, password))
    knowledge_graph = graph_from_neo4j()
    node_ids = list(knowledge_graph.nodes())
    edges_with_id = []
    positive_triples = []
    negative_triples = []
    for node1 in knowledge_graph.nodes:
        node1_info = knowledge_graph.nodes[node1]
        node1_name = node1_info["name"]
        node1_id = node1_info["nodeId"]
        if node1_info["defineName"] == "microorganism":
            query = "MATCH (givenNode)-[:CONNECTS*1..2]-(neighbor) WHERE givenNode.name = '" + node1_name + "' AND  neighbor.defineName = 'material' RETURN DISTINCT neighbor.name AS material_name, neighbor.nodeId AS nodeId"
            with driver.session() as session:
                result = session.run(query)
                for record in result:
                    material_name = record["material_name"]
                    material_id = record["nodeId"]
                    relation = 'isConnectedTo'
                    positive_triples.append((node1_name, relation, material_name))
                    edges_with_id.append((node1_id, material_id))
    num_negative_samples = len(positive_triples)
    for _ in range(num_negative_samples):
        node1_info = knowledge_graph.nodes[random.choice(node_ids)]
        node2_info = knowledge_graph.nodes[random.choice(node_ids)]
        relation = " "
        negative_triples.append((node1_info["name"], relation, node2_info["name"]))

    driver.close()
    return positive_triples, negative_triples, edges_with_id


def microorganism_material_adjacency_matrix():
    """
    微生物-材料节点的邻接矩阵
    :return: 返回微生物-材料的邻接矩阵
    """
    uri = "bolt://10.0.87.16:7687"
    username = "neo4j"
    password = "12345678"
    driver = GraphDatabase.driver(uri, auth=(username, password))
    all_ids = []
    microorganism_query = """MATCH (n)
                            WHERE n.defineName = 'microorganism'
                            RETURN n"""
    with driver.session() as session:
        microorganism_result = session.run(microorganism_query)
        microorganism_num = len(list(microorganism_result))
        for record in microorganism_result:
            node_id = record["n"].id
            all_ids.append(node_id)
    material_query = """MATCH (n)
                    WHERE n.defineName = 'material'
                    RETURN n"""
    with driver.session() as session:
        material_result = session.run(material_query)
        material_num = len(list(material_result))
        for record in material_result:
            node_id = record["n"].id
            all_ids.append(node_id)
    unique_all_ids = list(set(all_ids))
    id_mapping = nodes_ids_mapping(unique_all_ids)
    adjacency_matrix = np.zeros((microorganism_num + material_num, microorganism_num + material_num))
    positive_samples, negative_samples, edges_with_ids = cypher_neo4j_samples()
    print(len(edges_with_ids))
    for edge in edges_with_ids:
        node1, node2 = edge
        new_node1 = id_mapping.get(node1)
        new_node2 = id_mapping.get(node2)
        if new_node1 is not None and new_node2 is not None:
            adjacency_matrix[new_node1][new_node2] = 1
            adjacency_matrix[new_node2][new_node1] = 1
    return adjacency_matrix


def build_new_gcn_samples():
    """
    构造GCN和GAT模型的正负样本
    :return: 返回正负样本数量
    """
    # positive_triples, negative_triples, edges_with_id = cypher_neo4j_samples()
    adjacency_matrix = microorganism_material_adjacency_matrix()
    edge_index_pos = np.column_stack(np.argwhere(adjacency_matrix != 0))
    edge_index_pos = torch.tensor(edge_index_pos, dtype=torch.long)
    edge_index_neg = np.column_stack(np.argwhere(adjacency_matrix == 0))
    edge_index_neg = torch.tensor(edge_index_neg, dtype=torch.long)

    # 从正负样本中随机选取相同数量的边，形成平衡的训练集
    num_pos_edges_number = edge_index_pos.shape[1]
    selected_neg_edge_indices = torch.randint(high=edge_index_neg.shape[1], size=(num_pos_edges_number,),
                                              dtype=torch.long)
    edge_index_neg_selected = edge_index_neg[:, selected_neg_edge_indices]
    edg_index_all = torch.cat((edge_index_pos, edge_index_neg_selected), dim=1)
    y = torch.cat((torch.ones((edge_index_pos.shape[1], 1)),
                   torch.zeros((edge_index_neg_selected.shape[1], 1))), dim=0)  # 将所有y值设置为1,0
    # 将邻接矩阵转换为无向图（会将每条边复制成两条相反的边）
    edge_index = to_undirected(edge_index_pos)
    return edge_index_pos, edge_index_neg_selected, edge_index, y


def search_microbe_accession(query):
    """
    查找 GenBank Accession 号码
    :param query:需要查找的微生物
    :return:
    """
    Entrez.email = "liuyf@cnic.cn"
    handle = Entrez.esearch(db="nuccore", term=query, retmax=1)
    record = Entrez.read(handle)
    handle.close()
    if 'IdList' in record and record['IdList']:
        return record['IdList'][0]
    elif 'WebEnv' in record and 'QueryKey' in record:
        # 如果返回的是 WebEnv 和 QueryKey，则使用 efetch 获取 IdList
        handle = Entrez.efetch(db="nuccore", webenv=record['WebEnv'], query_key=record['QueryKey'], retmax=1)
        sub_record = Entrez.read(handle)
        handle.close()

        if 'IdList' in sub_record and sub_record['IdList']:
            return sub_record['IdList'][0]

    return None


def get_meta_path():
    """
    获取知识图谱中所有的元路径实例
    :return: 返回所有的元路径实例
    """
    # uri = "bolt://10.0.87.16:7687"
    # username = "neo4j"
    # password = "12345678"

    uri = "bolt://localhost:7687"
    username = "neo4j"
    password = "123456"

    driver = GraphDatabase.driver(uri, auth=(username, password))
    # 获取元路径实例
    meta_path_types = [["lineage", "microorganism", "synthesis_method", "material", "element", "material"],
                       ['synthesis_method', 'microorganism', 'microorganism', 'synthesis_method']]
    paths = []
    for meta_path in meta_path_types:
        with driver.session() as session:
            if len(meta_path) == 6:
                query = (
                    f"MATCH p = "
                    f"(A:{meta_path[0]})-[:CONNECTS]-"
                    f"(B:{meta_path[1]})-[:CONNECTS]-"
                    f"(C:{meta_path[2]})-[:CONNECTS]-"
                    f"(D:{meta_path[3]})-[:CONNECTS]-"
                    f"(E:{meta_path[4]})-[:CONNECTS]-"
                    f"(F:{meta_path[5]})"
                    f"RETURN p"
                )
            if len(meta_path) == 4:
                query = (
                    f"MATCH p = "
                    f"(A:{meta_path[0]})-[:CONNECTS]-"
                    f"(B:{meta_path[1]})-[:CONNECTS]-"
                    f"(C:{meta_path[2]})-[:CONNECTS]-"
                    f"(D:{meta_path[3]})"
                    f"RETURN p"
                )
            result = session.run(query)
            for path in result:
                paths.append(path)
    return paths


def generate_random_walk(meta_paths):
    """
    生成游走序列
    :return:
    """
    graph = graph_from_neo4j()
    start_node_type = ["microorganism", "synthesis_method"]
    start_node_list = []
    for node_item in graph.nodes(data=True):
        if "defineName" in node_item[1]:
            if node_item[1]["defineName"] in start_node_type:
                start_node_list.append(node_item[0])
        else:
            continue

    walks_per_meta_path = 20  # 生成随机游走序列
    walk_length = 50          # 每个游走序列的长度
    all_walks = []

    for meta_path in start_node_list:
        for _ in range(walks_per_meta_path):
            walk = []
            start_node = int(meta_path)
            walk.append(start_node)

            for _ in range(walk_length - 1):
                current_node = walk[-1]
                neighbors = list(graph.neighbors(current_node))
                if len(walk) > 1:
                    if len(neighbors) == 1:
                        # walk.append(neighbors[0])
                        break
                    if len(neighbors) > 1:
                        next_node = random.choice(neighbors)
                        while next_node == walk[-2]:
                            next_node = random.choice(neighbors)
                        walk.append(next_node)
                    else:
                        break
                else:
                    neighbors = list(graph.neighbors(current_node))
                    if len(neighbors) > 0:
                        walk.append(random.choice(neighbors))
                    else:
                        break
            all_walks.append(walk)

    return all_walks


def build_meta_path_samples():
    """
    构造meta_path正负样本
    :return:
    """
    # uri = "bolt://10.0.87.16:7687"
    # username = "neo4j"
    # password = "12345678"
    # driver = GraphDatabase.driver(uri, auth=(username, password))
    # meta_path_types = [["microorganism", "synthesis_method", "material"]]
    # meta_paths = []
    # for meta_path in meta_path_types:
    #     with driver.session() as session:
    #         query = (
    #             f"MATCH p = "
    #             f"(A:{meta_path[0]})-[:CONNECTS]-"
    #             f"(B:{meta_path[1]})-[:CONNECTS]-"
    #             f"(C:{meta_path[2]})"
    #             f"RETURN ID(A) as A_id, ID(B) AS B_id, ID(C) AS C_id"
    #         )
    #         result = session.run(query)
    #         for path in result:
    #             meta_paths.append(path)
    # 构造正负样本
    knowledge_graph = graph_from_neo4j()
    nodes_list = list(knowledge_graph.nodes())
    id_mapping = nodes_ids_mapping(nodes_list)
    train_positive_samples = []
    test_positive_samples = []
    train_negative_samples = []
    test_negative_samples = []
    df = pd.read_csv("../kg/output_file.csv")
    name_to_id = {node['name']: node_id for node_id, node in knowledge_graph.nodes(data=True)}
    # 根据'name1'和'name2'的值获取对应节点的ID
    df['name1_id'] = df['microorganism_name'].map(name_to_id)
    df['name2_id'] = df['material_name'].map(name_to_id)
    df['name3_id'] = df['synthesis_name'].map(name_to_id)
    df.dropna(subset=['name1_id'], inplace=True)

    train_data = df[df["year"] < 2020][["name1_id", "name2_id", "name3_id"]].values.tolist()
    test_data = df[df["year"] >= 2020][["name1_id", "name2_id", "name3_id"]].values.tolist()

    defineName_microorganism = find_nodes_by_define_name(knowledge_graph, "microorganism")
    print(len(defineName_microorganism))
    defineName_material = find_nodes_by_define_name(knowledge_graph, "material")
    print(len(defineName_material))
    defineName_synthesis = find_nodes_by_define_name(knowledge_graph, "synthesis method")
    print(len(defineName_synthesis))

    for train_item in train_data:
        microorganism_node = id_mapping.get(train_item[0])
        material_node = id_mapping.get(train_item[1])
        synthesis_node = id_mapping.get(train_item[2])
        if synthesis_node == None:
            continue
        else:
            train_positive_samples.append((microorganism_node, synthesis_node, material_node))

    for microorganism, synthesis, material in train_positive_samples:
        negative_microorganism = microorganism
        negative_material = material
        negative_synthesis = synthesis
        while (negative_microorganism, negative_synthesis, negative_material) in train_positive_samples:
            negative_microorganism = id_mapping.get(random.choice(defineName_microorganism))
            negative_material = id_mapping.get(random.choice(defineName_material))
            negative_synthesis = id_mapping.get(random.choice(defineName_synthesis))
        train_negative_samples.append((negative_microorganism, negative_synthesis, negative_material))

    for test_item in test_data:
        microorganism_node = id_mapping.get(test_item[0])
        material_node = id_mapping.get(test_item[1])
        synthesis_node = id_mapping.get(test_item[2])
        test_positive_samples.append((microorganism_node, synthesis_node, material_node))

    for microorganism, synthesis, material in test_positive_samples:
        negative_microorganism = microorganism
        negative_material = material
        negative_synthesis = synthesis
        while (negative_microorganism, negative_synthesis, negative_material) in test_positive_samples:
            negative_microorganism = id_mapping.get(random.choice(defineName_microorganism))
            negative_material = id_mapping.get(random.choice(defineName_material))
            negative_synthesis = id_mapping.get(random.choice(defineName_synthesis))
        test_negative_samples.append((negative_microorganism, negative_synthesis, negative_material))

    print(train_positive_samples)
    print(train_negative_samples)
    print(test_positive_samples)
    print(test_negative_samples)

    return train_positive_samples, train_negative_samples, test_positive_samples, test_negative_samples


def build_samples_MRR():
    """
    构造MRR数据集
    :return:
    """
    # 构造正负样本
    knowledge_graph = graph_from_neo4j()
    nodes_list = list(knowledge_graph.nodes())
    id_mapping = nodes_ids_mapping(nodes_list)
    train_positive_samples = []
    test_positive_samples = []
    train_negative_samples = []
    test_negative_samples = []
    df = pd.read_csv("../kg/output_file.csv")
    name_to_id = {node['name']: node_id for node_id, node in knowledge_graph.nodes(data=True)}
    # 根据'name1'和'name2'的值获取对应节点的ID
    df['name1_id'] = df['microorganism_name'].map(name_to_id)
    df['name2_id'] = df['material_name'].map(name_to_id)
    df.dropna(subset=['name1_id'], inplace=True)

    train_data = df[df["year"] < 2020][["name1_id", "name2_id"]].values.tolist()
    test_data = df[df["year"] >= 2020][["name1_id", "name2_id"]].values.tolist()

    defineName_microorganism = find_nodes_by_define_name(knowledge_graph, "microorganism")
    print(len(defineName_microorganism))
    defineName_material = find_nodes_by_define_name(knowledge_graph, "material")
    print(len(defineName_material))

    for train_item in train_data:
        microorganism_node = id_mapping.get(train_item[0])
        material_node = id_mapping.get(train_item[1])
        train_positive_samples.append((microorganism_node, material_node))

    for microorganism, material in train_positive_samples:
        negative_microorganism = microorganism
        negative_material = material
        while (negative_microorganism, negative_material) in train_positive_samples:
            negative_microorganism = id_mapping.get(random.choice(defineName_microorganism))
            negative_material = id_mapping.get(random.choice(defineName_material))
        train_negative_samples.append((negative_microorganism, negative_material))

    for test_item in test_data:
        microorganism_node = id_mapping.get(test_item[0])
        material_node = id_mapping.get(test_item[1])
        test_positive_samples.append((microorganism_node, material_node))

    for microorganism, material in test_positive_samples:
        negative_microorganism = microorganism
        negative_material = material
        while (negative_microorganism, negative_material) in test_positive_samples:
            negative_microorganism = id_mapping.get(random.choice(defineName_microorganism))
            negative_material = id_mapping.get(random.choice(defineName_material))
        test_negative_samples.append((negative_microorganism, negative_material))

    print(train_positive_samples)
    print(train_negative_samples)
    print(test_positive_samples)
    print(test_negative_samples)

    return train_positive_samples, train_negative_samples, test_positive_samples, test_negative_samples


def read_test_positive_samples():
    """
    读取到测试集中的正样本
    :return: 测试集中的正样本
    """
    df = pd.read_csv("../kg/output_file.csv")
    # 根据'name1'和'name2'的值获取对应节点的ID
    df['name1_id'] = df['microorganism_name']
    df['name2_id'] = df['material_name']
    df.dropna(subset=['name1_id'], inplace=True)

    # train_data = df[df["year"] < 2020][["name1_id", "name2_id"]].values.tolist()
    test_data = df[df["year"] >= 2020][["name1_id", "name2_id"]].values.tolist()
    return test_data


def find_nodes_by_define_name(graph, value):
    nodes = []
    for node in graph.nodes(data=True):
        if 'defineName' in node[1] and node[1]['defineName'] == value:
            # 如果节点满足条件，则将其添加到结果列表中
            nodes.append(node[0])  # node[0] 是节点的 ID
    return nodes


def get_name_by_id(graph, node_id):
    # 检查图中是否包含指定 ID 的节点
    if node_id in graph.nodes:
        # 获取节点对象
        node = graph.nodes[node_id]
        # 检查节点对象中是否包含 'name' 属性
        if 'name' in node:
            # 返回节点的 'name' 属性值
            return node['name']
        else:
            return None  # 如果节点不包含 'name' 属性，则返回 None
    else:
        return None  # 如果图中不包含指定 ID 的节点，则返回 None


def samples_by_year():
    """
    按年份划分正负样本
    :return: 返回正负样本
    """
    material_information_df = pd.read_csv("../kg/material_information.csv", usecols=['name', 'doi'])
    microorganism_information_df = pd.read_csv("../kg/microorganism_information.csv", usecols=['name', 'doi'])
    merged_df = pd.merge(material_information_df, microorganism_information_df, on='doi', how='inner')
    print(merged_df.head())
    # 打印合并后的DataFrame
    # print(merged_df)
    # for doi_item in doi_list:
    #     url = f'https://api.crossref.org/works/{doi_item}'
    # response = requests.get(url)
    # if response.status_code == 200:
    #     data = response.json()
    #     publication_date = data['message']['created']['date-time']
    #     return publication_date
    # else:
    #     return None


# 定义模型
class Classification(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Classification, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(0.5)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    print(0)
    # all_walks = generate_random_walk()
    # print(all_walks)
    # positive_samples, negative_samples = build_meta_path_samples()
    # samples = positive_samples + negative_samples
    # labels = [1] * len(positive_samples) + [0] * len(negative_samples)
    # print(len(positive_samples))
    # print(len(labels))
    # build_meta_path_samples()
    all_metapath = get_meta_path()
    for meta_item in all_metapath:
        print(meta_item)
