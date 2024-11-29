import argparse
import tempfile
import os
import streamlit as st
from eval_single import predict_text_from_audio


def process_wav_file(args, wav_file_path):
    st_message = predict_text_from_audio(args, wav_file_path)
    return st_message

# def page(args):
#     st.title("WAV File Chatbot")
#     uploaded_file = st.file_uploader("Choose a WAV file", type="wav")
#     with st.form(key='user_input_form'):
#         submit_button = st.form_submit_button(label='Start Recognizing')
#         if uploaded_file is not None and submit_button:
#             # 获取上传文件的父目录
#             wav_file_path = os.path.join("VCOP/data", uploaded_file.name)
#             st_message = process_wav_file(args, wav_file_path)
#             st.write("Output Message:")
#             st.write(st_message)

def page(args):
    st.title("WAV File Chatbot")
    uploaded_file = st.file_uploader("Choose a WAV file", type="wav")
    with st.form(key='user_input_form'):
        submit_button = st.form_submit_button(label='Start Recognizing')
        if submit_button:
            if uploaded_file is None:
                st.warning("Please upload a WAV file.")
            else:
                with st.spinner("Processing..."):
                    # 将上传的文件保存到临时文件
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                        temp_file.write(uploaded_file.getbuffer())
                        temp_file_path = temp_file.name
                    
                    # 获取临时文件的绝对路径
                    wav_file_path = os.path.abspath(temp_file_path)
                    
                    # 处理临时文件
                    st_message = process_wav_file(args, wav_file_path)
                    st.write("Output Message:")
                    st.write(st_message)
    

# if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default=None)
parser.add_argument('-n', '--ngpu', type=int, default=0)
parser.add_argument('-b', '--batch_size', type=int, default=4)
parser.add_argument('-bw', '--beam_width', type=int, default=5)
parser.add_argument('-p', '--penalty', type=float, default=0.6)
parser.add_argument('-ld', '--lamda', type=float, default=5)
parser.add_argument('-m', '--load_model', type=str, default='VCOP/save/model.pt')
parser.add_argument('-d', '--decode_set', type=str, default='test')
parser.add_argument('-ml', '--max_len', type=int, default=100)
parser.add_argument('-s', '--suffix', type=str, default=None)
args = parser.parse_args()

page(args)