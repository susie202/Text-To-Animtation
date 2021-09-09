# -*- coding: utf-8 -*-
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import socket
import sys
sys.path.append('waveglow/')
import numpy as np
import torch
from hparams import create_hparams
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser
from waveglow.mel2samp import MAX_WAV_VALUE
from text.symbols import symbols
import pickle
import struct
import yaml

def load_inference_model():
    tacotron_model_path = "./output/checkpoint_125000.pt"
    waveglow_model_path = './output/waveglow_54000'

    is_loaded = 1
    hparams = create_hparams()
    hparams.sampling_rate = 22050
    sampling_rate = hparams.sampling_rate
    ##tacotron2 model
    tacotron2 = load_model(hparams)
    tacotron2.load_state_dict(torch.load(tacotron_model_path)['state_dict'])
    _ = tacotron2.cuda().eval().half()
    ## waveglow model
    waveglow = torch.load(waveglow_model_path)['model']
    waveglow.cuda().eval().half()
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)

    return sampling_rate, tacotron2, waveglow, is_loaded

def inference(text, tacotron_model, waveglow_model):
    sequence = np.array(text_to_sequence(text, symbols))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

    mel_outputs, mel_outputs_postnet, _, alignments = tacotron_model.inference(sequence)

    with torch.no_grad():
        audio = waveglow_model.infer(mel_outputs_postnet, sigma=1.0)

    audio = audio * MAX_WAV_VALUE
    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    audio = audio.astype('int16')

    return audio

if __name__ == "__main__":
    ## load model
    sampling_rate, tacotron_model, waveglow_model, is_loaded = load_inference_model()
    print("Model loaded.")
    now_path = os.getcwd()
    host = ""
    port = 2048

    # host = "localhost"
    # port = 2048

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen()

    client_socket, addr = server_socket.accept()

    print("Client : ", addr)

    while 1:
        start = time.time()
        ##Input Text from client
        text = client_socket.recv(1024).decode()
        if text == "": break
        print("Text accepted.\n")

        ##Inference
        start_inference = time.time()
        np_audio = inference(text, tacotron_model, waveglow_model)
        end_inference = time.time()

        ##send array type
        np_send_data = pickle.dumps(np_audio)
        np_size = struct.pack("!I", len(np_send_data))

        #send size truncated
        client_socket.send(np_size)
        for i in range((len(np_send_data)//1024)+1):
            data = np_send_data[i * 1024: (i + 1) * 1024]
            client_socket.send(data)
        print("Wave sent. \n")
        end = time.time()
        with open("./Voice_Example/time.log", 'a', encoding="utf-8") as f:
            f.write("{} : {}second(Connect to Send) \t{}second(Inference Time) \n".format(text,end - start,end_inference - start_inference))
            f.close()

    print("Client disconnected")
    client_socket.close()
    server_socket.close()
