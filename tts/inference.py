#!/usr/bin/env python
# coding: utf-8

# ## Tacotron 2 inference code 
# Edit the variables **checkpoint_path** and **text** to match yours and run the entire code to generate plots of mel outputs, alignments and audio synthesis from the generated mel-spectrogram using Griffin-Lim.

# #### Import libraries and setup matplotlib

# In[1]:
text = "안녕하세요."
test_char = text
import time
# import matplotlib
# #from IPython import get_ipython
# #get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pylab as plt

import IPython.display as ipd

import sys
sys.path.append('waveglow/')
import numpy as np
import torch

from hparams import create_hparams
# from model import Tacotron2
# from layers import TacotronSTFT, STFT
# from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser


# In[ ]:


def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='lower',
                       interpolation='none')


# #### Setup hparams

# In[ ]:


hparams = create_hparams()
hparams.sampling_rate = 22050


# #### Load model from checkpoint

# In[ ]:


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

checkpoint_path = "./model/checkpoint_2500"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()


# #### Load WaveGlow for mel2audio synthesis and denoiser

# In[ ]:


waveglow_path = './model/waveglow_54000'
waveglow = torch.load(waveglow_path)['model']
waveglow.cuda().eval().half()
for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)


# In[ ]:

from text.symbols import symbols
# sequence = np.array(text_to_sequence(text, ['korean_cleaners']))[None, :]
sequence = np.array(text_to_sequence(text, symbols))[None, :]
sequence = torch.autograd.Variable(
    torch.from_numpy(sequence)).cuda().long()
print(sequence)
start_time = time.time()
mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)

# plot_data((mel_outputs.float().data.cpu().numpy()[0],
#            mel_outputs_postnet.float().data.cpu().numpy()[0],
#            alignments.float().data.cpu().numpy()[0].T))
#
# dump_mels = mel_outputs_postnet.cpu().detach().numpy()
# mels_save_path = os.path.join(os.getcwd(),'Voice_Example','{}_raw_mels.npy'.format(test_char))
# np.save(mels_save_path,dump_mels.astype(np.float32),allow_pickle=True);

from scipy.io.wavfile import write
from waveglow.mel2samp import MAX_WAV_VALUE
import torch.nn as nn

with torch.no_grad():
    audio = waveglow.infer(mel_outputs_postnet, sigma=1.0)
    audio = audio * MAX_WAV_VALUE
audio = audio.squeeze()
audio = audio.cpu().numpy()
audio = audio.astype('int16')
write("./Voice_Example/{}.wav".format(test_char), hparams.sampling_rate, audio)
infer_time = time.time() - start_time
print(infer_time)
with open("./Voice_Example/time.log",'a') as f:
    f.write("{} : {}second\n".format(test_char, infer_time))


