# 说明文档

### 1.安装环境

目前用的python版本是3.7，安装相关库时执行pip install requirements.txt

最后使用pip install PyAudio-0.2.11-cp37-cp37m-win_amd64.whl  安装pyaudio这个库

PyAudio-0.2.11-cp37-cp37m-win_amd64.whl在项目文件夹根目录下



如果是用的是gpu版本Asr_Gpu来运行此程序，那么你需要首先安装与电脑匹配的cuda11.1 和cudnn8.1.0，

另外将torch1.8.1，torchaudio替换为相应的gpu版本，再多安装一个相应gpu版本的torchvision第三方库。

### 2.数据集

本项目原本使用的语音数据集的是aishell-1和thchs30两个语音数据集，数据集文件是wav.rar文件，将数据集文件解压后放在根目录文件夹下，路径诸如Asr_Cpu/wav/xxx.wav。

### 3.运行程序

eval.py作用是批量测试训练集和测试集的字错误率

gui_show.py作用是界面，点击按钮后开是录音，再次点击按钮后结束录音，程序会自动识别出语音的结果并显示在界面中，该界面是人为可以控制录音时长

gui_show1.py作用是界面，点击按钮后开始录音，相隔一定时间后程序会自动结束录音，然后再次点击按钮后程序会自动识别出语音的结果并显示在界面中，该界面不可人为控制录音时长。录音时长参数设置在133行

```python
Record.recording(wave_name, RECORD_SECONDS=3)# RECORD_SECONDS 单位为秒
```

recog.py是推理单条语音识别结果的文件，以下这两行代码是设置模型加载的训练好的权重文件以及要识别的语音文件

```python
parser.add_argument('-m', '--load_model', type=str, default='egs/aishell/exp/transformer/model.epoch.191.pt')
parser.add_argument('-f', '--file', type=str, default='voices/caidan.wav')
```

record.py是与gui_show1.py配套的固定录音时间的代码

run.py是程序执行训练的代码，下面的'egs/aishell/conf/transformer.yaml'文件为训练参数的配置文件

```python
parser.add_argument('-c', '--config', type=str, default='egs/aishell/conf/transformer.yaml')
```

在程序中，训练前需要配置/修改的文件有以下几个：

'egs/aishell/conf/transformer.yaml'，模型参数的配置，具体需要自己配置

'egs/aishell/data/vocab'，训练集中出现的不重复的汉字，若要修改，按照该格式进行修改

'egs/aishell/data/train/'下面的character和wav.scp文件，分别是训练集的文本标签和训练集的语音文件名，若要修改，请按照原本格式进行修改

'egs/aishell/data/test/'下面的character和wav.scp文件，分别是测试集的文本标签和训练集的语音文件名，若要修改，请按照原本格式进行修改

### 4.其他注意的事项

原本的训练集使用的aishell-1和thchs30两个语音数据集音频特性分别是采样率16khz，单声道，如果你需要测试你自己的wav语音文件，那么请按照这两个特性进行录音。

如果你使用其他的数据集进行训练，那么也请保持所有数据集采样率和声道数的统一。程序中不需要修改采样率和声道数的参数，通过相关的库函数已经自适应不同音频特性的语音文件。