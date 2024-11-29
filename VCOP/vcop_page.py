import argparse
import tempfile
import os
import streamlit as st
from eval_single import predict_text_from_audio
import pyaudio
import wave
import sys
sys.path.append("..")
import utils
import time

# 设置参数
FORMAT = pyaudio.paInt16  # 音频格式
CHANNELS = 1  # 单声道
RATE = 16000  # 采样率
CHUNK = 1024  # 每个数据块的帧数
OUTPUT_FILENAME = "out.wav"  # 输出文件名
# 初始化pyaudio对象
p = pyaudio.PyAudio()
# Streamlit界面
st.title("你好, 我是AI空管助手小V！😃")
st.markdown("##")
col1, col2, col3 = st.columns([2, 6, 2])
with col1:
    st.image("src/imgs//half_pure_robot.png")
st.image("src/imgs//grass_crop.png")
# 按钮：开始录音
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'frames' not in st.session_state:
    st.session_state.frames = []


def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.1)

# 定义开始录音的函数
def start_recording():
    st.session_state.frames = []  # 清空之前录制的音频
    st.session_state.recording = True
    st.write("开始录音...")
    # # 打开音频流
    st.session_state.stream = p.open(format=FORMAT,
                                     channels=CHANNELS,
                                     rate=RATE,
                                     input=True,
                                     frames_per_buffer=CHUNK)
    
    # 录音过程
    while st.session_state.recording:
        data = st.session_state.stream.read(CHUNK)
        st.session_state.frames.append(data)

# 定义开始录音的函数
def start_recording_dummy():
    st.session_state.recording = True
    while st.session_state.recording:
        time.sleep(0.1)
    

# 定义停止录音的函数
def stop_recording():
    st.session_state.recording = False
    st.session_state.stream.stop_stream()
    st.session_state.stream.close()
    # 保存为wav文件
    with wave.open(OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(st.session_state.frames))
    st.write(f"成功录入！")

def stop_recording_dummy():
    st.session_state.recording = False
    time.sleep(0.2)
    st.write(f"成功录入！")

def process_wav_file(args, wav_file_path):
    st_message = predict_text_from_audio(args, wav_file_path)
    return st_message

def page(args):
    # st.title("WAV File Chatbot")
    with st.container(border=True):
        uploaded_file = st.file_uploader("请选择要识别的空管指令", type="wav")
        with st.form(key='user_input_form'):
            submit_button = st.form_submit_button(label='开始转译离线语音')
            if submit_button:
                if uploaded_file is None:
                    st.warning("Please upload a WAV file.")
                else:
                    with st.spinner("处理中..."):
                        # 将上传的文件保存到临时文件
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                            temp_file.write(uploaded_file.getbuffer())
                            temp_file_path = temp_file.name
                        # 获取临时文件的绝对路径
                        wav_file_path = os.path.abspath(temp_file_path)
                        # 处理临时文件
                        st_message = process_wav_file(args, wav_file_path)
                        st.write("Output:")
                        st.write_stream(stream_data(st_message))

    with st.container(border=True):
        st.write("或者使用麦克风：")
        # 控制按钮
        record_button = st.button("开始/停止")
        # utils.ChangeButtonColour('开始/停止', 'black', '#ACF4A2')
        utils.ChangeButtonColour('开始/停止', 'black', 'white')
        if record_button:
            if st.session_state.recording:
                # utils.ChangeButtonColour('开始/停止', 'black', '#ACF4A2')
                utils.ChangeButtonColour('开始/停止', 'black', 'white')
                stop_recording()
                # stop_recording_dummy()
            else:
                with st.spinner("录制中..."):
                    utils.ChangeButtonColour('开始/停止', 'black', '#FC5252')
                    start_recording()
                    # start_recording_dummy()
    
    with st.container(border=True):    
        tmp_record_button = st.button("开始转译在线语音...")
        if tmp_record_button:
            if os.path.isfile(OUTPUT_FILENAME):
                with st.spinner("Processing..."):
                    st_message = process_wav_file(args, "out.wav")
                    st.write("Output:")
                    st.write_stream(stream_data(st_message))
            else:
                st.warning("请先提供指令!")

# if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default=None)
parser.add_argument('-n', '--ngpu', type=int, default=1)
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