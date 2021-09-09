import sys
sys.path.append('./tts')

import torch
import numpy as np
from hparams import create_hparams
from train import load_model
from waveglow.denoiser import Denoiser
from waveglow.mel2samp import MAX_WAV_VALUE
from text import text_to_sequence
from text.symbols import symbols

def model_load(tacotron2_path='./models/checkpoint_125000.pt',waveglow_path='./models/waveglow_model/waveglow_54000'):
    ##tacotron2 model
    hparams = create_hparams()
    hparams.sampling_rate = 22050
    tacotron2 = load_model(hparams)
    tacotron2.load_state_dict(torch.load(tacotron2_path)['state_dict'])
    _ = tacotron2.cuda().eval().half()
    ## waveglow model
    waveglow = torch.load(waveglow_path)['model']
    waveglow.cuda().eval().half()
    for k in waveglow.convinv:
        k.float()
    _ = Denoiser(waveglow)
    return tacotron2, waveglow

def tacotron2_load(tacotron2_path='./models/checkpoint_125000.pt'):
    hparams = create_hparams()
    hparams.sampling_rate = 22050
    tacotron2 = load_model(hparams)
    tacotron2.load_state_dict(torch.load(tacotron2_path)['state_dict'])
    _ = tacotron2.cuda().eval().half()
    return tacotron2

def inference(text, tacotron2_model,waveglow_model):
    sequence = np.array(text_to_sequence(text, symbols))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()


    mel_outputs, mel_outputs_postnet, _, alignments = tacotron2_model.inference(sequence)

    with torch.no_grad():
        audio = waveglow_model.infer(mel_outputs_postnet, sigma=1.0)

    audio = audio * MAX_WAV_VALUE
    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    audio = audio.astype('int16')

    return audio

