import streamlit as st
import replicate
from PIL import Image
import io
from dataclasses import dataclass
from typing import Optional, Union
import base64
import utils
from gpt_copilot.main import web_warning_detect
import os, json, base64
from sensetime_llm import get_access_key, encode_jwt_token
import requests
from gpt_copilot.main import text2speech
import time

@dataclass
class ModelParams:
    text_prompt: str
    st_uploaded_img: Optional[st.runtime.uploaded_file_manager.UploadedFile]
    selected_model: str
    top_p: float
    max_tokens: int
    temperature: float
    replicate_link: str #only useful for those use service from replicate.com

def generate_llm_response(params: ModelParams) -> Union[str, dict]:
    if params.st_uploaded_img is not None:
        img_bytes = params.st_uploaded_img.getvalue()
        image = io.BytesIO(img_bytes)
        ## Convert image to base64
        # you can create a data URI consisting of the base64 encoded data for your file, but this is only recommended if the file is < 1mb:
        base64_encoded_image = base64.b64encode(img_bytes).decode('utf-8')
        base64_image_str = f"data:application/octet-stream;base64,{base64_encoded_image}"

    else:
        raise ValueError("Image must be provided for the selected model.")
    
    if params.selected_model == "llava-13B":
        
        output = replicate.run(
            # "yorickvp/llava-13b:80537f9eead1a5bfa72d5ac6ea6414379be41d4d4f6679fd776e9535d1eb58bb",
            params.replicate_link,
            input={
                "image": base64_image_str,#params.st_uploaded_img, #image, #base64_image_str
                "top_p": params.top_p,
                "prompt": params.text_prompt,
                "max_tokens": params.max_tokens,
                "temperature": params.temperature
            }
        )
    # DONE: Implement the logic for the "gpt-4o" model   
    elif params.selected_model == "gpt-4o":
        output = web_warning_detect(image)
    elif params.selected_model == 'sensetime-vision':
        ak, sk = get_access_key()
        API_TOKEN = encode_jwt_token(ak, sk)
        
        ## following params are from official web, you can change them as needed
        data = {
            "model": "SenseChat-Vision",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_base64",
                            "image_base64": base64_encoded_image
                        },
                        {
                            "type": "text",
                            "text": params.text_prompt
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
        json_data = json.dumps(data)
        # 定义请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_TOKEN}"
        }
        # 发送 POST 请求
        response = requests.post("https://api.sensenova.cn/v1/llm/chat-completions", headers=headers, data=json_data)
        # print(response.status_code)
        # print(response.json())
        output = response.json()["data"]["choices"][0]["message"]

    elif params.selected_model == "Qwen-vl":
        input = {
            "image": params.st_uploaded_img,
            "prompt": params.text_prompt,
        }

        output = replicate.run(
            params.replicate_link,
            # "lucataco/qwen-vl-chat:50881b153b4d5f72b3db697e2bbad23bb1277ab741c5b52d80cd6ee17ea660e9",
            input=input
        )
    else:
        raise NotImplementedError
    return output

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]


# st.set_page_config(page_title="✈️ Virtual Copilot 💬", layout="wide")

# 创建标签页
# tab1, tab2 = st.tabs(["Main", "Subpage"])
# 获取当前屏幕的宽度和高度
screen_height, screen_width = utils.get_screen_HW()
col1, col2, col3 = st.columns([0.25,0.5,0.25])
with col2:
    st.title('✈️ Virtual Copilot Assistant💬')
    st.caption("🚀 An AI-Assistant powered by MLLM")
    with st.sidebar:
        st.title('✈️ Virtual Copilot 💬')
        selected_model = st.sidebar.selectbox('Choose a foundation model', ['gpt-4o', 'sensetime-vision', 'llava-13B', 'Qwen-vl'], key='selected_model')

        st.subheader('Models and parameters')        
        if selected_model == 'Qwen-vl':
            replicate_link = 'lucataco/qwen-vl-chat:50881b153b4d5f72b3db697e2bbad23bb1277ab741c5b52d80cd6ee17ea660e9'
        elif selected_model == 'llava-13B':
            replicate_link = 'yorickvp/llava-13b:80537f9eead1a5bfa72d5ac6ea6414379be41d4d4f6679fd776e9535d1eb58bb'
        # the following models use official service directly, no need for replicate's link
        elif selected_model == 'gpt-4o':
            replicate_link = None
        elif selected_model == 'sensetime-vision':
            replicate_link = None            
        
        temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.2, step=0.01)
        top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
        max_tokens = st.sidebar.slider('max_tokens', min_value=64, max_value=4096, value=512, step=8)
        st.sidebar.button('Clear Chat History', on_click=clear_chat_history, type='primary')
        # st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)!')
        lab_text = utils.set_text('📖 Learn more about our [lab!](https://shineergo.wixsite.com/homepage)', font_size=16, font_weight='bold')
        st.markdown(lab_text, unsafe_allow_html=True)
        
    model_params = ModelParams(
        text_prompt=None,
        st_uploaded_img=None,
        selected_model=selected_model,
        top_p=top_p,
        max_tokens=max_tokens,
        temperature=temperature,
        replicate_link=replicate_link
    )
    # print(model_params.temperature)
    #Ensure session state for messages if it doesn't exist
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    if "promt_text" not in st.session_state:
        st.session_state.promt_text = ""

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("type") == "image":
                st.image(message["content"])
            else:
                st.write(message["content"])


    # Create a form for user input and file upload
    upload_img = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    with st.form(key='user_input_form'):
        if upload_img is not None:
            # print(upload_img, "uploaded", type(upload_img))
            image = Image.open(upload_img)
            st.image(image, caption='Uploaded Image', use_column_width=True)
        text_input = st.text_input("Type a message:", key="input_text")
        submit_button = st.form_submit_button(label='Send')


    if submit_button:
        prompt = text_input
        if upload_img:
            #If an image is uploaded, store it in session_state
            st.session_state.messages.append({"role": "user", "content": upload_img, "type": "image"})

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.prompt_text = ""  # Clear the input text

        model_params.text_prompt = prompt
        model_params.st_uploaded_img = upload_img

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = generate_llm_response(model_params)
                    # response = ["Use dummpy reply for test"]
                    placeholder = st.empty()
                    full_response = ''
                    for item in response:
                        full_response += item
                        placeholder.markdown(full_response)
                    # placeholder.markdown(full_response)

            text2speech(full_response)
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)
            # if os.path.exists("tmp.mp3"):
                # st.audio('tmp.mp3', format="audio/mpeg", loop=True)

            # wait for io complete in text2speech
            timeout = 2
            start_time = time.time()
            while time.time() - start_time < timeout:
                if os.path.exists("tmp.mp3") and os.path.getsize("tmp.mp3") > 0:
                    st.audio("tmp.mp3", format="audio/mpeg", loop=False)
                    break
                time.sleep(0.1)  
        
