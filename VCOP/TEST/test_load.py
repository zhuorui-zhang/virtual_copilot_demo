import torchaudio
import os
assert os.path.exists(r"G:\audio\2024-06-01T13-24-03.413453.wav"), "文件路径无效！"

print(str(torchaudio.list_audio_backends()))
# 加载音频文件
waveform, sample_rate = torchaudio.load("G:\\audio\\2024-06-01T13-24-03.413453.wav")

print(waveform)