import random

# f = open('filelists/script_spl.txt', 'r')
# lines = f.readlines()
# random.shuffle(lines)
# f.close()
#
# train_txt = 'filelists/train.txt'
# valid_txt = 'filelists/valid.txt'
#
#
# for idx, line in enumerate(lines):
#     if idx < 12900:
#         fileName = train_txt
#     else:
#         fileName = valid_txt
#
#     fw = open(fileName, 'a')
#     fw.write(line)
#     fw.close()

import wave
import matplotlib.pyplot as plt

# 13000 in total
f = open('filelists/script_spl.txt', 'r')
lines = f.readlines()
f.close()

# 12241 in total
txt_name = 'filelists/script_10s.txt'
wav_path = '/media/user/samsung_1TB/DB/synthesis/01.Main/raw/'

count = 0
duration_list = []
for idx, line in enumerate(lines):
    filename = wav_path + line.split('\t')[0] + '.wav'
    audio = wave.open(filename)
    frames = audio.getnframes()
    rate = audio.getframerate()
    duration = frames / float(rate)
    duration_list.append(duration)
    if duration <= 10:
        count += 1
        # fw = open(txt_name, 'a')
        # fw.write(line)
        # fw.close()

print(count)
print(idx+1)
plt.hist(duration_list, bins=100)
plt.show()