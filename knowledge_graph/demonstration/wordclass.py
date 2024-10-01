import spacy
from collections import Counter
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #正常显示负号
# 加载英语模型
import os

def fun_wordclass(file_path,wordclass_output_path, show_plot=True):
    curren_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(curren_dir, file_path)
    wordclass_output_path = os.path.join(curren_dir, wordclass_output_path)
    nlp = spacy.load("en_core_web_sm")
    # 指定文件路径
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        text = file.read()
    # 使用spacy处理文本
    doc = nlp(text)
    # 提取所有词性
    tags = [token.pos_ for token in doc]
    # 统计词性频率
    tag_counts = Counter(tags)
    # 绘制词性统计图
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(tag_counts.keys(), tag_counts.values(), color='darkorange')
    ax.set_xlabel('词性')
    ax.set_ylabel('数量')
    ax.set_title('词性统计')
    # Rotating X-axis labels
    ax.set_xticklabels(tag_counts.keys(), rotation = 45)

    pos_words = defaultdict(list)
    # 提取所有词性及其对应的单词
    for token in doc:
        pos_words[token.pos_].append(token.text)
    # 将词性及其对应的单词转换为 DataFrame
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in pos_words.items()]))
    print(df)

    # 导出 DataFrame 为 Excel 文件
    excel_file_path = wordclass_output_path
    df.to_excel(excel_file_path, index=False)
    print(f"词性统计数据已导出到 {excel_file_path}")
    if show_plot:
        plt.show()
    else:
        return fig