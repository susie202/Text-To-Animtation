import random

f = open('filelists/script_10s.txt', 'r')
lines = f.readlines()
random.shuffle(lines)
f.close()

train_txt = 'filelists/train.txt' #12140 in total
valid_txt = 'filelists/valid.txt' #  100 in total


for idx, line in enumerate(lines):
    if idx < 12140:
        fileName = train_txt
    else:
        fileName = valid_txt

    fw = open(fileName, 'a')
    fw.write(line)
    fw.close()