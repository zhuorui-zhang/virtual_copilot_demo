
import time
import jwt
import csv
import requests
import base64
import json
from utils import print_red

def get_access_key():
    ### Download your access key from https://console.sensecore.cn/iam/Security/access-key
    file_path = '密钥.csv' 
    ak_column = 'Access key ID'  # 替换为您要读取的列名
    sk_column = 'AccessKey Secret'
    ak, sk = None, None
    try:
        with open(file_path, mode='r', encoding='utf-8-sig') as file:
            csv_reader = csv.DictReader(file, skipinitialspace=True)
            for row in csv_reader:
                ak = row[ak_column]
                sk = row[sk_column]
                break
    except FileNotFoundError:
        print_red("Please download your access key from https://console.sensecore.cn/iam/Security/access-key")

    return ak, sk

def encode_jwt_token(ak, sk):
    headers = {
        "alg": "HS256",
        "typ": "JWT"
    }
    payload = {
        "iss": ak,
        "exp": int(time.time()) + 1800, # 填写您期望的有效时间，此处示例代表当前时间+30分钟
        "nbf": int(time.time()) - 5 # 填写您期望的生效时间，此处示例代表当前时间-5秒
    }
    api_token = jwt.encode(payload, sk, headers=headers)
    return api_token

    
if __name__ == "__main__":
    ak, sk = get_access_key()
    API_TOKEN = encode_jwt_token(ak, sk)
    # print(API_TOKEN) # 打印生成的API_TOKEN

    # # 查看模型列表，你也可以参考https://platform.sensenova.cn/doc?path=/platform/helpdoc/help.md (文档中心)
    # # 定义 API URL 和头部信息
    # url = "https://api.sensenova.cn/v1/llm/models"
    # headers = {
    #     "Authorization": f"Bearer {API_TOKEN}",
    #     "Content-Type": "application/json"
    # }
    # # 发送 GET 请求
    # response = requests.get(url, headers=headers)

    # # 打印响应内容
    # # print(response.status_code)
    # # print(response.json())

    # # curl --request GET "https://api.sensenova.cn/v1/llm/models" \
    # #   -H "Authorization: Bearer $API_TOKEN" \
    # #   -H "Content-Type: application/json"

    ###-----------------------------------------------------------------

    # 读取本地图片并转换为 base64
    with open("test.jpg", "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    # 定义 JSON 数据
    data = {
        "model": "SenseChat-Vision",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_base64",
                        "image_base64": image_base64
                    },
                    {
                        "type": "text",
                        "text": "string"
                    }
                ]
            }
        ],
        "max_new_tokens": 1024,
        "repetition_penalty": 1.05,
        "stream": False,
        "temperature": 0.5,
        "top_p": 0.25,
        "user": "string"
    }

    # 将 JSON 数据转换为字符串
    json_data = json.dumps(data)

    # 定义请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_TOKEN}"
    }
    # 发送 POST 请求
    response = requests.post("https://api.sensenova.cn/v1/llm/chat-completions", headers=headers, data=json_data)
    # 打印输出
    # print(response.status_code)
    print(response.json())


    ##----------------------------------------------------------------
    # # import subprocess
    # # import base64
    # # import json
    # # # 读取本地图片并转换为 base64
    # # with open("test.jpg", "rb") as image_file:
    # #     image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    # # # 定义 JSON 数据
    # # data = {
    # #     "model": "SenseChat-Vision",
    # #     "messages": [
    # #         {
    # #             "role": "user",
    # #             "content": [
    # #                 {
    # #                     "type": "image_base64",
    # #                     "image_base64": image_base64
    # #                 },
    # #                 {
    # #                     "type": "text",
    # #                     "text": "string"
    # #                 }
    # #             ]
    # #         }
    # #     ],
    # #     "max_new_tokens": 1024,
    # #     "repetition_penalty": 1.05,
    # #     "stream": False,
    # #     "temperature": 0.5,
    # #     "top_p": 0.25,
    # #     "user": "string"
    # # }

    # # # 将 JSON 数据转换为字符串
    # # json_data = json.dumps(data)

    # # # 定义 curl 命令
    # # curl_command = [
    # #     "curl", "--request", "POST", "https://api.sensenova.cn/v1/llm/chat-completions",
    # #     "-H", "Content-Type: application/json",
    # #     "-H", f"Authorization: Bearer {API_TOKEN}",
    # #     "-d", "@-"
    # # ]

    # # # 运行 curl 命令并通过标准输入传递 JSON 数据
    # # process = subprocess.Popen(curl_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # # stdout, stderr = process.communicate(input=json_data)

    # # # 打印输出
    # # print(stdout)
    # # print(stderr)

