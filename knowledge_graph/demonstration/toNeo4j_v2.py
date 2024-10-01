
from py2neo import Graph, NodeMatcher, Node, Relationship
import subprocess
import webbrowser
import pandas as pd
import platform, os
import time, requests

# def wait_for_neo4j_startup(url, timeout=60):
#     """轮询检查 Neo4j 是否已经启动并统计等待时长"""
#     start_time = time.time()
#     while time.time() - start_time < timeout:
#         try:
#             response = requests.get(url)
#             if response.status_code == 200:
#                 elapsed_time = time.time() - start_time
#                 print(f"Neo4j 已成功启动，等待时长: {elapsed_time:.2f} 秒")
#                 return True
#         except requests.ConnectionError:
#             pass
#         time.sleep(0.1)
#     raise TimeoutError("等待 Neo4j 启动超时")

def wait_for_neo4j_startup(url, bolt_url, user, password, timeout=60):
    """轮询检查 Neo4j 是否已经启动并统计等待时长"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                # 尝试连接并执行一个简单的 Cypher 查询
                try:
                    graph = Graph(bolt_url, auth=(user, password))
                    graph.run("RETURN 1")
                    elapsed_time = time.time() - start_time
                    print(f"Neo4j 已成功启动，等待时长: {elapsed_time:.2f} 秒")
                    return True
                except Exception as e:
                    print(f"Neo4j HTTP 接口已启动，但数据库未完全准备好: {e}")
        except requests.ConnectionError:
            pass
        time.sleep(0.1)
    raise TimeoutError("等待 Neo4j 启动超时")

# for linux platform
def is_neo4j_running():
    try:
        result = subprocess.run(['neo4j', 'status'], capture_output=True, text=True, check=True)
        return 'running' in result.stdout.lower()
    except subprocess.CalledProcessError:
        return False
    
def connectNeo4j_fun():
    # 打开默认浏览器并访问 http://localhost:7474/
    # this is disabled because it will open a new browser window
    # and the user may not want that
    # webbrowser.open("http://localhost:7474/")
    
    # 判断系统平台
    system_platform = platform.system()

    if system_platform == "Windows":
        # 执行命令 neo4j.bat console
        subprocess.Popen("neo4j.bat console", shell=True)
    elif system_platform == "Linux" or system_platform == "Darwin":
        if is_neo4j_running():
            subprocess.run(['neo4j', 'stop'], check=True)
        # subprocess.run("neo4j console", shell=True, check=True)
        subprocess.Popen("neo4j console", shell=True)
    else:
        raise OSError("Unsupported operating system")
    
    print("Neo4j已启动")

def connect_fun(user,password):
    # 连接Neo4j数据库输入地址、用户名、密码

    # 发现linux平台下链接http会出现一个py2neo的error (不影响使用)
    # url = 'http://localhost:7474/browser/'
    url = 'bolt://localhost:7687'
    graph = Graph(url, auth = (user,password))
    matcher = NodeMatcher(graph) #创建关系需要用到
    print("连接成功")
    return graph,matcher

def delAll_fun(graph):
    query = "MATCH (n) DETACH DELETE n"
    graph.run(query)
    print('删除所有节点')

def creat_crewman(df_crewman,defined_nodes,graph, matcher):
    for index, row in df_crewman.iterrows():  #定义action
        crew = str(row['crew'])
        crew_node=None
        # age=str(row['note'])
        if crew in defined_nodes:
            crew_node=graph.nodes.match("Crewman", name=crew).first()
        else:
            # crew_node=Node('Crewman',name=crew,age=age)
            crew_node = Node('Crewman', name=crew)
            defined_nodes.add(crew)
            graph.create(crew_node)
    return defined_nodes

def creat_feedback(df_feedback,defined_nodes,graph, matcher):
    for index, row in df_feedback.iterrows():#定义feedback
        X=str(row['ActionResult'])
        try:
            Y=str(row['By'])
        except:
            Y='pilot'
        relationship=str(row['feedback'])
        # 检查节点是否已定义，如果已定义，则直接使用已定义的节点
        if X in defined_nodes:
            X_node = graph.nodes.match("ActionResult", name=X).first()
        else:
            X_node = Node('ActionResult', name=X)
            defined_nodes.add(X)
            graph.create(X_node)
        if Y in defined_nodes:
            Y_node = graph.nodes.match(name=Y).first()  #只根据name索引
            Y_node.add_label('By')
            graph.push(Y_node)
        else:
            Y_node= Node('By', name=Y)
            defined_nodes.add(Y)
            graph.create(Y_node)
        # 创建操作方式关系
        relationship = Relationship(X_node,relationship, Y_node)
        graph.create(relationship)
    return defined_nodes

def creat_action(df_action,defined_nodes,graph, matcher):
    for index, row in df_action.iterrows():
        X = str(row['ActionReason'])
        Y = str(row['ActionObject'])
        relation= str(row['Action'])
        # 检查节点是否已定义，如果已定义，则直接使用已定义的节点
        if X in defined_nodes:
            X_node = graph.nodes.match(name=X).first()
            X_node.add_label('ActionReason')
            graph.push(X_node)
        else:
            X_node = Node('ActionReason', name=X)
            defined_nodes.add(X)
            graph.create(X_node)
        if Y in defined_nodes:
            Y_node = graph.nodes.match(name=Y).first()  # 只根据name索引
            Y_node.add_label('ActionObject')
            graph.push(Y_node)
        else:
            Y_node = Node('ActionObject', name=Y)
            defined_nodes.add(Y)
            graph.create(Y_node)
        # 创建操作方式关系
        relationship = Relationship(X_node, relation, Y_node)
        graph.create(relationship)


    for index, row in df_action.iterrows():
        X = str(row['ActionObject'])
        Y = str(row['ActionResult'])
        relationship = 'result in'
        # 检查节点是否已定义，如果已定义，则直接使用已定义的节点
        if X in defined_nodes:
            X_node = graph.nodes.match("ActionObject", name=X).first()
        else:
            X_node = Node('ActionObject', name=X)
            defined_nodes.add(X)
            graph.create(X_node)
        if Y in defined_nodes:
            Y_node = graph.nodes.match(name=Y).first()  # 只根据name索引
            Y_node.add_label('ActionResult')
            graph.push(Y_node)
        else:
            Y_node = Node('ActionResult', name=Y)
            defined_nodes.add(Y)
            graph.create(Y_node)
        # 创建操作方式关系
        relationship = Relationship(X_node, relationship, Y_node)
        graph.create(relationship)
    return defined_nodes

def creat_event(df_event,defined_nodes,graph, matcher):
    for index, row in df_event.iterrows():
        X = str(row['event'])
        Y = str(row['subtask'])
        try:
            relation = str(row['type'])
        except:
            relation = 'subtask'
        # 检查节点是否已定义，如果已定义，则直接使用已定义的节点
        if X in defined_nodes:
            X_node = graph.nodes.match(name=X).first()
            X_node.add_label('event')
            graph.push(X_node)
        else:
            X_node = Node('event', name=X)
            defined_nodes.add(X)
            graph.create(X_node)
        if Y in defined_nodes:
            Y_node = graph.nodes.match(name=Y).first()  # 只根据name索引
            Y_node.add_label('subtask')
            graph.push(Y_node)
        else:
            Y_node = Node('subtask', name=Y)
            defined_nodes.add(Y)
            graph.create(Y_node)
        # 创建操作方式关系
        relationship = Relationship(X_node, relation, Y_node)
        graph.create(relationship)
    return defined_nodes

def creat_condition(df_condition,defined_nodes,graph, matcher):
    for index, row in df_condition.iterrows():
        X = str(row['condition'])
        Y = str(row['causedBy'])
        relation ='causedBy'
        # 检查节点是否已定义，如果已定义，则直接使用已定义的节点
        if X in defined_nodes:
            X_node = graph.nodes.match(name=X).first()
            X_node.add_label('condition')
            graph.push(X_node)
        else:
            X_node = Node('condition', name=X)
            defined_nodes.add(X)
            graph.create(X_node)
        if Y in defined_nodes:
            Y_node = graph.nodes.match(name=Y).first()  # 只根据name索引
            Y_node.add_label('condition')
            graph.push(Y_node)
        else:
            Y_node = Node('condition', name=Y)
            defined_nodes.add(Y)
            graph.create(Y_node)
        # 创建操作方式关系
        relationship = Relationship(X_node, relation, Y_node)
        graph.create(relationship)
    return defined_nodes

def gen_graph(file, graph, matcher):
    print('开始创建')
    data = pd.read_excel(file, sheet_name=None)
    df_action = data['action']
    df_feedback = data['feedback']
    df_event = data['event']
    df_crewman = data['crewman']
    df_condition = data['condition']
    # 存储已定义的节点名称
    defined_nodes = set()
    # 定义crewman
    defined_nodes = creat_crewman(df_crewman, defined_nodes, graph, matcher)
    # 定义feedback
    defined_nodes = creat_feedback(df_feedback, defined_nodes, graph, matcher)
    # 定义action
    defined_nodes = creat_action(df_action, defined_nodes, graph, matcher)
    # 定义event
    defined_nodes = creat_event(df_event, defined_nodes, graph, matcher)
    # 定义condition
    defined_nodes = creat_condition(df_condition, defined_nodes, graph, matcher)
    print('创建成功')

    
def main(file_path):
    user = "neo4j"
    # password = '180823'
    password = 'abc12345'
    #注：创建、删除操作在neo4j中有1~2秒的延迟
    connectNeo4j_fun()
    # import time
    # time.sleep(5)
    url = "http://localhost:7474/browser/"
    bolt_url = "bolt://localhost:7687"
    wait_for_neo4j_startup(url, bolt_url, user, password, timeout=60)
    graph, matcher=connect_fun(user,password)
    delAll_fun(graph)
    
    gen_graph(file_path, graph, matcher) 
    
if __name__=="__main__":
    xiangsici_path = "knowledge_graph/demonstration/90-FMS Engine Failure Procedures/新知识合并结果.xlsx"
    main(xiangsici_path)   



