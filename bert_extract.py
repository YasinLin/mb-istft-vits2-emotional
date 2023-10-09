import re

import numpy as np

from prosody import TTSProsody
from pypinyin import lazy_pinyin, BOPOMOFO
import torch
import sys
import jieba
from transformers import AutoTokenizer, AutoModelForMaskedLM
from text.mandarin import number_to_chinese, chinese_to_bopomofo, latin_to_bopomofo,_latin_to_bopomofo
from vits_bert import get_vits_bert

def preprocess_one(path, text):
    emb = get_vits_bert().chinese_to_bert(text)
    path = path.replace(".wav", ".bert.npy")
    np.save(f"{path}", emb, allow_pickle=False)
    return emb


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Bert Extraction Preprocess')
    parser.add_argument('--filelists', dest='filelists',nargs="+", type=str, help='path of the filelists')
    args = parser.parse_args()

    for filelist in args.filelists:
        print(filelist,"----start bert extract-------")
        with open(filelist) as f:
            for idx, line in enumerate(f.readlines()):
                arr = line.strip().split("|")
                path, text = arr[0],arr[2]
                preprocess_one(path,text)
