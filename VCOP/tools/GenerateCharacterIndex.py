data = [
    "CES6503上升到5700米",
    "CES3021下降到1800米",
    "CSN2586下降到2100米",
    "CSN3569下降到2400米",
    "CDG1356下降到2700米",
    "CDG1472下降到3100米",
    "CDG1483下降到3400米",
    "CDG1493下降到3700米",
    "CES6203下降到4000米",
    "CES6350下降到4300米",
    "CES6430下降到4600米",
    "CES6430下降到4900米",
    "CES6430下降到5200米",
    "银川雷达看到",
    "直飞三角区",
    "建立盲降",
    "飞航向90",
    "飞航向180",
    "飞航向270",
    "飞航向360",
    "左转90度",
    "右转90度",
    "再见",
    "起飞",
    "减速到180",
    "减速到200",
    "加速到300",
    "加速到400",
    "减速到150",
    "减速到100"
]

# flatten the list to a single string
text = ''.join(data)

# extract all unique characters
characters = sorted(set(text))

# print the dictionary without colon and with single space as a delimiter
for i, char in enumerate(characters, start=1):
    print(f'{char} {i}')



for i in range(1, 31):  # assuming the files are numbered from 1 to 30
    print(f"a{i}.mp3 F:\\speech recognition\\Asr_Gpu_ATC\\wav\\a{i}.mp3")
