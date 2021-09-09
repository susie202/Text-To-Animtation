import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse
import glob
import csv
import numpy as np
from config.AudioConfig import AudioConfig
from inference import predict

def prepare_inference(wav_path, img_path):
    command = 'python scripts/prepare_testing_files.py --src_audio_path \"{}\" --src_input_path path \"{}\"'.format(wav_path, img_path)
    os.system(command)


if __name__ == "__main__":
    emotion = 'hap'
    wav_path = "misc/Audio_Source/demo_voice.mp3"
    character = 1
    
    img_path = os.path.join(str(character), emotion)
    #img_path = os.path.join('face-generation', str(character), emotion)
    print(img_path)
    predict()


    
    # character랑 emotion으로 os.path 만들어주자
    #path = os.path()
    
    
    
