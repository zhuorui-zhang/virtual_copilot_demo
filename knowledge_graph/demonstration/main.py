import pandas as pd

from wordclass import fun_wordclass
# from toNeo4j_v2 import connectNeo4j_fun,connect_fun,delAll_fun,gen_graph
import toNeo4j_v2
import os

def demo_func(file_path=None):

    # 第一步：词性统计
    print('*'*10+'第一步：词性统计'+'*'*10)
    wordclass_output_path='wordclass_output_path.xlsx'  #词汇统计数据导出路径
    fun_wordclass(file_path,wordclass_output_path)

    #第二步:三元组显示
    print('*'*10+'第二步:三元组显示'+'*'*10)
    sanyuanzu_path='三元组.xlsx'
    df_sanyuanzu=pd.read_excel(sanyuanzu_path)
    print(df_sanyuanzu)

    #第三步：相似词合并
    print('*'*10+'第三步：相似词合并'+'*'*10)
    xiangsici_path = "90-FMS Engine Failure Procedures/新知识合并结果.xlsx"
    excel_file = pd.ExcelFile(xiangsici_path)
    # 获取所有工作表的名称
    sheet_names = excel_file.sheet_names
    # 打印工作表名称及其内容
    for sheet in sheet_names:
        print(f'Sheet name: {sheet}')
        # 读取当前工作表的内容
        df = excel_file.parse(sheet)
        # 打印当前工作表的内容
        print(df)
        print('\n')
    #第四步：生成知识图谱
    print('*'*10+'第四步：生成知识图谱'+'*'*10)
    # connectNeo4j_fun()
    # usr = 'neo4j'
    # key = '180823'
    # graph, matcher=connect_fun(usr,key)
    # delAll_fun(graph)
    # file=r"90-FMS Engine Failure Procedures/新知识合并结果.xlsx"
    # gen_graph(file,graph, matcher)

    toNeo4j_v2.main(file_path = xiangsici_path)
    
if __name__=="__main__":
    os.chdir(os.path.dirname(__file__))
    file_path = r"90-FMS Engine Failure Procedures/FMS Engine Failure Procedures .txt"
    demo_func(file_path)