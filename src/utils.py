
import re
import glob
import json
import torch

from sklearn.model_selection import train_test_split
from datasets import load_dataset

from overlap import prepro

def get_train_e3c():
    path_train = "../../corpus/E3C-French/splits/split_1"
    train = f"{path_train}/train"
    val = f"{path_train}/val"
    textes = []
    for fic in glob.glob(f"{train}/*.txt"):
        with open(fic, "r", encoding="utf-8") as fin:
            contenu = fin.read()
            textes.append(contenu)
    for	fic in glob.glob(f"{val}/*.txt"):
       	with open(fic, "r", encoding="utf-8") as fin:
       	    contenu = fin.read()
       	    textes.append(contenu)
    textes = [f"<|startoftext|> {d[0:]} <|endoftext|>" for d in textes]
    return textes

def lire_e3c():
    chemin = "../../corpus/E3C-French/fichiers_json"
    contenu = []
    total_tokens = 0
    for fic in glob.glob("%s/*" % chemin):
        with open(fic, "r", encoding="utf-8") as fin:
            dic = json.load(fin)
            if dic["type"] in ["journal", "pubmed"]:
                nb_toks = len(dic["text"].split())
                total_tokens += nb_toks
                contenu.append(dic["text"])
    contenu = [f"<|startoftext|> {d[0:-1]} <|endoftext|>" for d in contenu] 
    return contenu

def lire_e3c_annotations():
    chemin = "../../corpus/E3C-French/fichiers_xml"
    contenu = []
    for fic in glob.glob("%s/*" % chemin):
        with open(fic, "r", encoding="utf-8") as fin:
            contenu.append(fin.read())
    contenu = [f"<|startoftext|> {d[0:-1]} <|endoftext|>" for d in contenu]
    return contenu		

def lire_e3c_for_overlap():
    corpus_file = {}
    corpus_text = ""
    chemin = "../../corpus/E3C-French/fichiers_json"
    contenu = []
    total_tokens = 0
    for fic in glob.glob("%s/*" % chemin):
        with open(fic, "r", encoding="utf-8") as fin:
            dic = json.load(fin)
            if dic["type"] in ["journal", "pubmed"]:
                fname = fic.split("/")[-1]
                cleaned = prepro.prepro(dic["text"])
                corpus_file[fname] = f" \n {cleaned}"
                corpus_text += f" \n {cleaned}"
    return corpus_file, corpus_text


def data_collator(features):
    first = features[0]
    batch = {}
    for k, v in first.items():
        batch[k] = torch.tensor([f[k] for f in features])
    return batch

def load_special_tokens(bos_and_eos: bool, path_special_tokens=None):
    special_tokens = []
    if path_special_tokens is not None:
        with open("liste_special_tokens.json", "r", encoding="utf-8") as fin:
            special_tokens = json.load(fin)
    if bos_and_eos == True:
        special_tokens += ["<|startoftext|>", "<|endoftext|>"]
    return special_tokens
