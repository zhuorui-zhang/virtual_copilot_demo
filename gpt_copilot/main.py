import pandas as pd  # for DataFrames to store article sections and embeddings
import numpy as np
import openai  # for generating embeddings
import os  # for environment variables
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image,ImageDraw
import base64
import io
import requests
import tiktoken
import json
import re
import time
import cv2
import json

from  get_bndbox import get_boxes
from rsg import rsg

import easyocr
import matplotlib.pyplot as plt

SAVE_PATH = "img_instructed_warning_dataset2_0725.csv"
data_path = r'data/merged0518.csv'
model="gpt-4o"
#model = "gpt-4-turbo-preview"
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))

img_analyze_prompt = '''
    You are a pilot and you will be provided with a describe of a panel in the cabin in flight.It carries aircraft parameters and warning messages.\
    Your role is to recognize and list warnings from the description.
'''

img_prompt_base = ''' 
Find the red and yellow warnings displayed on the left side of the the warning display.think step by step.Output the result with brace,for example:

{EMERGENCY OR ABNORMAL 1}
{EMERGENCY OR ABNORMAL 2}
...
{EMERGENCY OR ABNORMAL N}

If there are no such warnings, you should only output a {} .

'''

img_prompt_cot = ''' 
Firstly, find the warnings displayed on the left side of the warning display. Do NOT split the warning messages.

Secondly, identify the unindented warnings on the left side.

Then,determine the color of each warning message on the left side, line by line.

Finally, output the red and yellow warnings on the left side that are not indented .You should output them with braces,for example:

{EMERGENCY OR ABNORMAL 1}
{EMERGENCY OR ABNORMAL 2}
...
{EMERGENCY OR ABNORMAL N}

If there are no such warnings, you should only output a {} . 

'''


system_prompt = '''
    You are a co-pilot and your role is to give instructions to the captain in case of an emergency. \
    I will give you the type of emergency and the related solution from a quick reference book, and you will return the correct answer based on these context. 

    If the content is not relevant with the emergency,you should tell me why it is not relevant and tell me you can not find related solution.
    You should stay concise with your answer, replying specifically to the input prompt without mentioning additional information. 
    Any mistake may result in an air crash.
'''


def count_tokens_in_message(message: str) -> int:
    # 加载 GPT-4 Turbo 模型的编码器
    encoding = tiktoken.encoding_for_model("gpt-4-turbo")
    # 将消息编码为 tokens
    tokens = encoding.encode(message)
    # 返回 token 的数量
    return len(tokens)


def encode_image(image):
    buffer = io.BytesIO()
    # 将图像以某种格式保存到缓冲区
    image.save(buffer, format="JPEG")  # 确保与实际图像文件的格式匹配
    # 获取缓冲区的二进制内容
    binary_data = buffer.getvalue()
    # 关闭缓冲区
    buffer.close()
    return base64.b64encode(binary_data).decode('utf-8')



def bndbox_analyze(base64_image,bnd_box):
    api_key = os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    num=len(bnd_box)
    payload = {
        "model": "gpt-4-vision-preview",  # gpt-4-vision-preview
        "messages": [
            {
                "role": "system",
                "content": f"recognize the warning text in the image and then only return the warning"
            },

            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    },

                ]
            }
        ],
        "max_tokens": 1000,
        "top_p": 0.1
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    data = json.loads(response.text)
    # 提取 "content" 字段的值
    content = data['choices'][0]['message']['content']
    return content


def get_embeddings(text):
    embeddings = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
        encoding_format="float"
    )
    return embeddings.data[0].embedding


def search_content(df, input_text, top_k, obj_name='embedding_emergency'):
    embedded_value = get_embeddings(input_text)
    if obj_name == 'embedding_actions':
        df["similarity"] = df.embedding_actions.apply(
            lambda x: cosine_similarity(np.array(x).reshape(1, -1),
                                        np.array(embedded_value).reshape(1, -1)))  # embedding_emergency
    else:
        df["similarity"] = df.embedding_emergency.apply(
            lambda x: cosine_similarity(np.array(x).reshape(1, -1),
                                        np.array(embedded_value).reshape(1, -1)))  # embedding_emergency
    res = df.sort_values('similarity', ascending=False).head(top_k)
    return res


def get_similarity(row):
    similarity_score = row['similarity']
    if isinstance(similarity_score, np.ndarray):
        similarity_score = similarity_score[0][0]
    return similarity_score


def find_last_number(sentence):
    # Use regex to find all numbers in the sentence
    numbers = re.findall(r'\d+', sentence)

    # Check if there are any numbers found
    if numbers:
        # Return the last number in the list
        return numbers[-1]
    else:
        return None


def generate_output(input_prompt, similar_content, threshold=0.99):
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
                   f" then identify the one that most closely matches the warning message {input_prompt} .Finally,return the reason and corresponding serial number,for example 1"
                   f" The out put should follow this format:reason:here is the reason,id:here is the corresponding serial number")
        start_time = time.time()
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
        elapsed_time = time.time() - start_time
        id = completion1.choices[0].message.content[-1]
        if not id.isdigit():
            id = find_last_number(completion1.choices[0].message.content)
        content = similar_content.iloc[int(id) - 1]['actions']
        count1 = count_tokens_in_message(prompt1)

    prompt2 = (
        f"INPUT PROMPT:\nGive instructions to pilot under emergency of {input_prompt} based on the content.\n-------\nCONTENT:\n{content},\ "
        f"If the result contain many steps,you should give the instruction step by step.For example,First,...,Secondly,...Then...")
    count2 = count_tokens_in_message(prompt2) + count_tokens_in_message(system_prompt)
    start_time = time.time()
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
    end_time = time.time()
    elapsed_time += (end_time - start_time)
    token_num = count1 + count2
    return completion.choices[0].message.content, token_num, elapsed_time


def generate_output_cp(input_prompt, similar_content, threshold=0.99):
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
        content = ""
        for i, row in similar_content.iterrows():
            content += f"\n{row['actions']}"
    # prompt2 = (f"INPUT PROMPT:\nGive instructions to pilot under emergency of {input_prompt} based on the content.\n-------\nCONTENT:\n{content},\ "
    #          f"If the result contain many steps,you should give the instruction step by step.For example,First,...,Secondly,...Then...")
    prompt2 = (
        f"INPUT PROMPT:\nGive instructions to pilot under emergency of {input_prompt} based on the content.\n-------\nCONTENT:\n{content},\ "
        f"If the result contain many steps,you should give the instruction step by step.For example,First,...,Secondly,...Then...")
    count2 = count_tokens_in_message(prompt2) + count_tokens_in_message(system_prompt)
    start_time = time.time()
    completion = client.chat.completions.create(
        model=model,
        temperature=0.2,
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
    end_time = time.time()
    elapsed_time += (end_time - start_time)
    token_num = count1 + count2
    return completion.choices[0].message.content, token_num, elapsed_time


def format_string(input_string):
    # 去除字符串两端的空白字符并按换行符分割字符串
    lines = input_string.strip().split('\n')

    # 遍历每一行并去除两端的空白字符
    formatted_lines = [line.strip() for line in lines]

    return formatted_lines


def extract_brace_content(text):
    # 使用正则表达式提取花括号中的内容
    pattern = re.compile(r'\{([^}]*)\}')
    matches = pattern.findall(text)
    return matches


def contains_when_or_if(input_string):
    # Convert the string to uppercase to ensure case-insensitive search
    input_string_upper = input_string.upper()

    # Check if 'WHEN' or 'IF' is in the string
    if 'WHEN' in input_string_upper or 'IF' in input_string_upper:
        return True
    else:
        return False


def extract_bndboxes_from_file(json_path):
    with open(json_path, 'r') as file:
        json_data = json.load(file)

    bndboxes = []
    roi_box = None
    objects = json_data["outputs"]["object"]
    for obj in objects:
        if isinstance(obj, list):
            for ob in obj:
                bndboxes.append(ob["bndbox"])
        else:
            if obj["name"]=="roi":
                roi_box=obj["bndbox"]
            else:
                bndboxes.append(obj["bndbox"])
    return bndboxes,roi_box


if __name__ == '__main__':
        image_dir = r"data/test_img"  # 替换为实际图像路径186
        img_files = [file for file in os.listdir(image_dir) if file.endswith('.jpg')]
        img_detect = 1
        table = pd.DataFrame()
        i=0
        for img_file in img_files:
            image_path = os.path.join(image_dir, img_file)
            print(img_file)
            img = Image.open(image_path)
            # 设置缩放比例或目标尺寸
            scale_factor = 0.5  # 缩小为原始尺寸的 50%
            new_width = int(img.width * scale_factor)
            new_height = int(img.height * scale_factor)
            # 使用 Image 类的 resize 方法进行下采样
            resampled_img = img.resize((new_width, new_height))

            if img_detect:
                warnings, roi= get_boxes(img_file, resampled_img)
                size = {"width": new_width, "height": new_height, "depth": 3}
                flag=True if len(warnings)>0 else False
                data = {
                    "path": img_file,
                    "outputs": {
                        "object": [roi, warnings]
                    },
                    "time_labeled": int(time.time() * 1000),  # 当前时间的毫秒数
                    "labeled": flag,
                    "size": size
                }

                # 将数据转换为JSON格式字符串
                json_data = json.dumps(data, indent=4)
                # 保存到文件
                image_name = os.path.splitext(img_file)[0]
                json_filename = f"{image_name}.json"
                with open(json_filename, "w") as json_file:
                    json_file.write(json_data)
                print(json_filename+" processed successfully")
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

            json_file=os.path.join(os.path.splitext(img_file)[0] + '.json')
            bnd_ls,roi=extract_bndboxes_from_file(json_file)
            if len(roi)==0:
                results = 'No warning'
            else:
                xmin=int(0.85*roi["xmin"])
                ymin=int(0.95*roi["ymin"])
                xmax=int(roi["xmax"])
                ymax=int(roi["ymax"])
                crop_img = resampled_img.crop((xmin,ymin,xmax,ymax))
                roi_box_gt = [xmin,ymin, xmax, ymax]
                roi_box_gt2 = [xmin,ymin,xmin,ymin]
                resampled_img = crop_img.resize((512, 512))
                factor1=512/(roi_box_gt[2]-roi_box_gt[0])#1024
                factor2=512/(roi_box_gt[3]-roi_box_gt[1])
                msg_box_ls=[]
                draw = ImageDraw.Draw(resampled_img)
                result='No warning'
                results=[]
                if len(bnd_ls)>0:
                    for json_box in bnd_ls:
                        msg_bnd_box = [json_box["xmin"], json_box["ymin"], json_box["xmax"], json_box["ymax"]]
                        msg_rlt_box = [max(0,msg_bnd_box[j] - roi_box_gt2[j]) for j in range(4)]
                        msg_rlt_box_cp = msg_rlt_box

                        msg_box = {"left top corner x": int(msg_rlt_box[0] * factor1),
                                   "left top corner y": int(msg_rlt_box[1] * factor2),
                                   "right bottom corner x": int(msg_rlt_box[2] * factor1),
                                   "right bottom corner y": int(msg_rlt_box[3] * factor2)}
                        msg_box_ls.append(msg_box)
                        rect=resampled_img.crop((int(msg_rlt_box[0] * factor1),int(msg_rlt_box[1] * factor2),
                                                 int(msg_rlt_box[2] * factor1),int(msg_rlt_box[3] * factor2)))
                        draw.rectangle([(msg_box["left top corner x"],msg_box["left top corner y"]),(msg_box["right bottom corner x"],msg_box["right bottom corner y"])], outline='green',width=4)
                        #rect.show()
                        base64_image = encode_image(rect)
                        result = bndbox_analyze(base64_image, msg_box_ls)
                        warning=re.findall(r'"(.*?)"', result)
                        results.append(warning[0])
                        print(results)
            reply=rsg(results[0], data_path)
            print(reply)
















