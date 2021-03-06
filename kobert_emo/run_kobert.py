import os
import logging
import argparse
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoModelForSequenceClassification

from kobert_emo.utils import init_logger, load_tokenizer

import kss
import re
# from hanspell import spell_checker
from soynlp.normalizer import repeat_normalize


logger = logging.getLogger(__name__)

# model_dir = 'kobert_emo/model'


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_args():
    return torch.load('./models/kobert_emo/training_args.bin')


def model_load(model_dir='./model'):
    device = "cuda"
    # Check whether model exists
    if not os.path.exists(model_dir):
        raise Exception("Model doesn't exists! Train first!")

    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)# Config will be automatically loaded from model_dir
        model.to(device)
        model.eval()
        logger.info("***** Model Loaded *****")
    except:
        raise Exception("Some model files might be missing...")

    return model

def convert_input_file_to_tensor_dataset(input, args,
                                         cls_token_segment_id=0,
                                         pad_token_segment_id=0,
                                         sequence_a_segment_id=0,
                                         mask_padding_with_zero=True):
    tokenizer = load_tokenizer(args)

    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []

    tokens = tokenizer.tokenize(input)
    # Account for [CLS] and [SEP]
    special_tokens_count = 2
    if len(tokens) > args.max_seq_len - special_tokens_count:
        tokens = tokens[:(args.max_seq_len - special_tokens_count)]

    # Add [SEP] token
    tokens += [sep_token]
    token_type_ids = [sequence_a_segment_id] * len(tokens)

    # Add [CLS] token
    tokens = [cls_token] + tokens
    token_type_ids = [cls_token_segment_id] + token_type_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = args.max_seq_len - len(input_ids)
    input_ids = input_ids + ([pad_token_id] * padding_length)
    attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
    token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

    all_input_ids.append(input_ids)
    all_attention_mask.append(attention_mask)
    all_token_type_ids.append(token_type_ids)

    # Change to Tensor
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)

    return dataset


#0814 ?????? ????????? ?????? ??????  
#Preprocess input text
def clean_text(text):

    """
    ?????? ?????????
    """
    pattern = re.compile(
        r'[^ .,?!/@$%~????????()\x00-\x7F???-???]+')
    url_pattern = re.compile(
        r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

    text = pattern.sub(' ', text)
    text = url_pattern.sub(' ', text)
    text = re.sub(r'\.\.+', '.', text) # ?????? ??? ??? ?????? ???????????? ?????? ??????
    text = re.sub(r'\u200B', '', text) # ??? ?????? ?????? ??????
    text = re.sub('???', '', text) # ??? ??????
    text = re.sub('#', ' ', text) # tab ??????
    text = re.sub('\t', ' ', text)
    text = text.strip()
    text = repeat_normalize(text, num_repeats=3)

    # ?????? ?????? ..??? ?????? ?????? .?????? ??????
    while re.search('[???-???]\.\.\s', text):
        text = re.sub(r'\.\.', '\.', text)

    # ?????? ?????? ?????? ????????? ??? .. ??? ?????? ?????? ??????
    text = re.sub(r'\.\.', '', text) 

    return text

def sentence_split(text):
    sen = []
    for sent in kss.split_sentences(text):
        sen.append(sent)
    result = ' '.join(sen)
#     result = '. '.join(sen)
    return result


# def spell_check(text):
#     if len(text) >= 500:
#         pass
#     else:
#         text = spell_checker.check(text).as_dict()["checked"]
#     return text

def preprocess(sen):
    sen = clean_text(sen)
    sen = sentence_split(sen)
    # sen = spell_check(sen)
    return sen



#model_dir??????.
def predict(input, model):
    # load model and args
    args = get_args()
    device = get_device()


    #0814 ?????? ????????? ??????
    # Preprocessing
    input = preprocess(input)

    # Convert input file to TensorDataset
    dataset = convert_input_file_to_tensor_dataset(input, args)

    # Predict
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=32)

    preds = None

    for batch in tqdm(data_loader, desc="Predicting"):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": None}
            if args.model_type != "distilkobert":
                inputs["token_type_ids"] = batch[2]
            outputs = model(**inputs)
            logits = outputs[0]

            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=1)

    return preds
