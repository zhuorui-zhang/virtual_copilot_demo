from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import os
import pandas as pd
from scipy import sparse
import openai  # for generating embeddings
import tiktoken
import pickle
from ast import literal_eval

model="gpt-4o"
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))


def count_tokens_in_message(message: str) -> int:
    # 加载 GPT-4 Turbo 模型的编码器
    encoding = tiktoken.encoding_for_model("gpt-4-turbo")
    # 将消息编码为 tokens
    tokens = encoding.encode(message)
    # 返回 token 的数量
    return len(tokens)


# clean text
def clean_text(text):
    # 去除多余空格
    cleaned_text = re.sub(r'\s+', ' ', text)
    # 去除空白行
    cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)
    return cleaned_text.strip()


def extract_titles_from_txt(txt_path):
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            title = lines[0].strip() if len(lines) > 0 else None  # 提取第一行作为标题
            content = ' '.join(lines[1:]).strip() if len(lines) > 1 else None  # 提取action
            return title, content
    except Exception as e:
        print(f"Error processing {txt_path}: {e}")
        return None, None


def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text

system_prompt = '''
    You are a co-pilot and your role is to give instructions to the captain in case of an emergency. \
    I will give you the type of emergency and the related solution from a quick reference book, and you will return the correct answer based on these context. 

    If the content is not relevant with the emergency,you should tell me why it is not relevant and tell me you can not find related solution.
    You should stay concise with your answer, replying specifically to the input prompt without mentioning additional information. 
    Any mistake may result in an air crash.
'''


def find_last_number(sentence):
    # Use regex to find all numbers in the sentence
    numbers = re.findall(r'\d+', sentence)

    # Check if there are any numbers found
    if numbers:
        # Return the last number in the list
        return numbers[-1]
    else:
        return None


def generate_output_pr(input_prompt, similar_content, threshold=0.99):
    token_num = 0
    elapsed_time = 0
    n = 0
    count1 = 0
    count2 = 0
    if similar_content.iloc[0]['similarity'] > threshold:
        content = similar_content.iloc[0]['actions']
        emergency = similar_content.iloc[0]['emergency']
    # Adding more matching content if the similarity is above threshold
    else:
        emergency = ""
        for i, row in similar_content.iterrows():
            emergency += f"\n\n{n + 1}:{row['emergency']}"
            n = n + 1
        prompt1 = (f"you should first analyze each operational title below {emergency} ,\ "
                   f" then identify the one that most closely matches the warning message {input_prompt} ."
                   f"Finally,return the reason and corresponding serial number,for example 1")
        completion1 = client.chat.completions.create(
            model=model,
            temperature=0.5,
            messages=[

                {
                    "role": "user",
                    "content": prompt1
                }
            ]
        )
        id = completion1.choices[0].message.content[-1]
        if not id.isdigit():
            id = find_last_number(completion1.choices[0].message.content)
        content = similar_content.iloc[int(id) - 1]['actions']
        count1 = count_tokens_in_message(prompt1)

    prompt2 = (
        f"INPUT PROMPT:\nGive instructions to pilot under emergency of {input_prompt} based on the content.\n-------\nCONTENT:\n{content},\ "
        f"If the result contain many steps,you should give the instruction step by step.For example,First,...,Secondly,...Then...")
    count2 = count_tokens_in_message(prompt2) + count_tokens_in_message(system_prompt)
    completion = client.chat.completions.create(
        model=model,
        temperature=0.5,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt2
            },

        ]
    )
    token_num = count1 + count2
    return completion.choices[0].message.content, token_num, elapsed_time



def generate_output(input_prompt,similar_titles, similar_content, threshold=0.99):
    token_num=0
    elapsed_time=0
    n=0


    prompt2 = (f"INPUT PROMPT:\nGive instructions to pilot under emergency of {input_prompt} based on the content.\n-------\nCONTENT:\n{similar_content},\ "
               f"If the result contain many steps,you should give the instruction step by step.For example,First,...,Secondly,...Then...")

    completion = client.chat.completions.create(
        model=model,
        temperature=0.5,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt2
            },

        ]
    )
    count2 = count_tokens_in_message(prompt2)+count_tokens_in_message(system_prompt)
    token_num=count2
    return completion.choices[0].message.content, token_num, elapsed_time


def get_embeddings(text):
    embeddings = client.embeddings.create(
      model="text-embedding-3-small",
      input=text,
      encoding_format="float"
    )
    return embeddings.data[0].embedding


def search_content(df, input_text, top_k=500, obj_name='embedding_emergency'):
    embedded_value = get_embeddings(input_text)
    if obj_name == 'embedding_actions':
        df["similarity"] = df.embedding_actions.apply(
            lambda x: cosine_similarity(np.array(x).reshape(1, -1), np.array(embedded_value).reshape(1, -1)))#embedding_emergency
    else:
        df["similarity"] = df.embedding_emergency.apply(
            lambda x: cosine_similarity(np.array(x).reshape(1, -1), np.array(embedded_value).reshape(1, -1)))#embedding_emergency
    if top_k >= len(df):
        res = df.sort_values('similarity', ascending=False)
    else:
        res = df.sort_values('similarity', ascending=False).head(top_k)
    return res


def interleave_top_k_half(method1, method2,contents1,contents2, k=5):
    top_k_half_1 = method1
    top_k_half_2 = method2
    c_top_k_half_1 = contents1
    c_top_k_half_2 = contents2
    interleaved_results = []
    contents=[]
    for i in range(k):
        if i < len(top_k_half_1) and top_k_half_1[i] not in interleaved_results:
            interleaved_results.append(top_k_half_1[i])
            contents.append(c_top_k_half_1[i])
        if i < len(top_k_half_2) and top_k_half_2[i] not in interleaved_results:
            interleaved_results.append(top_k_half_2[i])
            contents.append(c_top_k_half_2[i])

    return interleaved_results[:k],contents[:k]


def rsg(exs,data_path):
    instruction = ''
    dff = pd.read_csv(data_path, encoding='ISO-8859-1')
    dff["embedding_emergency"] = dff.embedding_emergency.apply(literal_eval).apply(np.array)
    dff["embedding_actions"] = dff.embedding_actions.apply(literal_eval).apply(np.array)
    if exs == "nan" or exs == "" or exs == ",":
        return instruction
    if isinstance(exs, float):
        return instruction
    split_strings = [s.strip() for s in exs.split(',') if s.strip()]
    ex = split_strings[0]
    ex = ex.replace("\n", "")
    matching_content_em = search_content(dff, ex)
    cc = matching_content_em.emergency.index
    matching_content5_em = search_content(dff, ex, 5)
    instruction, token_num_pr_5, elapsed_time = generate_output_pr(ex, matching_content5_em)
    return instruction

