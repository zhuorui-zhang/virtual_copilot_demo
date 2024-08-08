#20240618张振生
import pandas as pd
import matplotlib.pyplot as plt
from py2neo import Graph, NodeMatcher
import networkx as nx
from py2neo import Graph, NodeMatcher, Node, Relationship
import networkx as nx
# 设置显示所有行和列

from toNeo4j import connectNeo4j_fun,connect_fun
# connectNeo4j_fun()
def connect_fun(usr,key):
    url = 'http://localhost:7474/browser/'
    graph = Graph(url,auth = (usr,key))
    matcher = NodeMatcher(graph) #创建关系需要用到
    print("连接成功")
    return graph,matcher
usr = 'neo4j'
key = '180823'
graph, matcher = connect_fun(usr, key)

q1 = "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 20"
q1 = "MATCH (n)-[r]->(m) RETURN n, r, m "
data = graph.run(q1).data()

# 创建空的NetworkX图
G = nx.DiGraph()
# 添加节点和边到NetworkX图
for record in data:
    n = record['n']
    m = record['m']
    r = record['r']
    G.add_node(n.identity, label=n['name'])
    G.add_node(m.identity, label=m['name'])
    G.add_edge(n.identity, m.identity, relationship=r.__class__.__name__)
# 自动调整窗口大小
num_nodes = G.number_of_nodes()
figsize = (max(10, num_nodes * 2), max(7, num_nodes * 1.5))  # 根据节点数动态调整窗口大小
# 绘制图形
pos = nx.spring_layout(G)  # 节点布局
plt.figure(figsize=(10, 7))
nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'), node_size=2000, node_color='skyblue',
        font_size=10, font_color='black', font_weight='bold', edge_color='grey')
plt.title('Neo4j Graph Visualization')
plt.show()