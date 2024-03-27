import sys
import os

from sklearn.linear_model import SGDClassifier, LinearRegression, Lasso, Ridge
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
import seaborn as sn
import random
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from sklearn.manifold import TSNE
import tqdm
from transformers import RobertaTokenizer, RobertaModel

import copy
from sklearn.svm import LinearSVC

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import TruncatedSVD
import torch

from sklearn.linear_model import SGDClassifier

from sklearn.svm import LinearSVC
from pytorch_revgrad import RevGrad

import sklearn
from sklearn.linear_model import LogisticRegression
import random
import pickle
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import neural_network
import numpy as np
from transformers import BertModel, BertTokenizer,DebertaTokenizer , DebertaModel
import argparse
import time
import wandb
from collections import Counter

import torch.nn.functional as F
import math

DATA_PATH = "/your_path/data/"

if True:
    parser = argparse.ArgumentParser(description="An argparse example")
    parser.add_argument('--model', type=str, default="deberta-base")
    parser.add_argument('--run_id', type=int, default=0, required=False)
    parser.add_argument('--device', type=int, default=0, required=False)
    parser.add_argument('--emb_mode', type=int, default=-1, required=False)
    parser.add_argument('--lam_end', type=float, default=0, required=False)
    parser.add_argument('--dataset', type=str, default="moji", required=False)
    args = parser.parse_args()
    print(args)



    lambda_threshold = args.lam_end
    pre_defined = (args.emb_mode == 0)
    data_driven = (args.emb_mode == 1)
    data_learned = (args.emb_mode == 2)
    emb_mode_name = "reg" if args.emb_mode < 0 else "pd" if pre_defined else "dd" if data_driven else "dl" if data_learned else "donno"

    path = f"models/{args.dataset}/{args.model}/reg/{args.run_id}"

random.seed(args.run_id)
np.random.seed(args.run_id)
torch.manual_seed(args.run_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

embedding_dim = 768
if args.model == "bert":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model_version = "bert-base-uncased"

elif args.model == "deberta-base":
    tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
    model = DebertaModel.from_pretrained('microsoft/deberta-base', output_attentions=False,
                                        output_hidden_states=False)
    model_version = "deberta-base"

elif args.model == "roberta":
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')
    model_version = "roberta-base"
else:
    raise ValueError("Model not supported")

def balance_data(input_ids, attention_masks, y, z):
        ratio = 0.7
        n = 100000

        happy_aa_idx = np.where(np.logical_and(z == 1, y == 1))[0]
        happy_sa_idx = np.where(np.logical_and(z == 0, y == 1))[0]
        sad_aa_idx = np.where(np.logical_and(z == 1, y == 0))[0]
        sad_sa_idx = np.where(np.logical_and(z == 0, y == 0))[0]

        smallest_subset_len1 = min(len(sad_sa_idx), len(happy_aa_idx))
        alternative_n1 = smallest_subset_len1 / (ratio / 2)

        smallest_subset_len2 = min(len(happy_sa_idx), len(sad_aa_idx))
        alternative_n2 = smallest_subset_len2 / ((1 - ratio) / 2)

        n = min(min(n, alternative_n1), alternative_n2)

        n_1 = int(n * ratio / 2)
        n_2 = int(n * (1 - ratio) / 2)

        all_indices = []
        for sub_dataset, amount in [(happy_aa_idx, n_1), (happy_sa_idx, n_2), (sad_aa_idx, n_2), (sad_sa_idx, n_1)]:
            perm = np.random.permutation(len(sub_dataset))
            idx = sub_dataset[perm[:amount]]
            all_indices.extend(idx)

        return input_ids[all_indices], attention_masks[all_indices], y[all_indices], z[all_indices]

def load_bios(group):
    path = DATA_PATH + "biasbios/"
    X = np.load(path+"{}_cls.npy".format(group))
    with open(path+"{}.pickle".format(group), "rb") as f:
        bios_data = pickle.load(f)
        Y = np.array([1 if d["g"] == "f" else 0 for d in bios_data])
        professions = np.array([d["p"] for d in bios_data])
        txts = [d["hard_text_untokenized"] for d in bios_data]
        random.seed(0)
        np.random.seed(0)
        X, Y, professions, txts, bios_data = sklearn.utils.shuffle(X, Y, professions, txts, bios_data)
        X = X[:]
        Y = Y[:]

    return X, Y, txts, professions, bios_data
def tokenize_data(dataset, tokenizer, model_version, path):
    if dataset == "bios":
        max_length = 128
        X, y_gender, txts, professions, bios_data = load_bios("train")
        X_dev, y_dev_gender, txts_dev, professions_dev, bios_data_dev = load_bios("dev")
        X_test, y_test_gender, txts_test, professions_test, bios_data_test = load_bios("test")

        X, y_gender, txts, professions = X[:], y_gender[:], txts[:], professions[:]
        X_dev, y_dev_gender, txts_dev, professions_dev = X_dev[:], y_dev_gender[:], txts_dev[:], professions_dev[:]
        X_test, y_test_gender, txts_test, professions_test = X_test[:], y_test_gender[:], txts_test[
                                                                                          :], professions_test[:]

        prof2ind = {p: i for i, p in enumerate(sorted(set(professions)))}
        ind2prof = {i: p for i, p in prof2ind.items()}
        y_prof = np.array([prof2ind[p] for p in professions])
        y_dev_prof = np.array([prof2ind[p] for p in professions_dev])
        y_test_prof = np.array([prof2ind[p] for p in professions_test])

        encoded_dict = tokenizer(txts, add_special_tokens=True, padding='max_length', max_length=max_length,
                             truncation=True, return_attention_mask=True)

        input_ids = encoded_dict['input_ids']
        attention_masks = encoded_dict['attention_mask']

        input_ids = np.array(input_ids)
        attention_masks = np.array(attention_masks)
        y = np.array(y_prof)
        z = np.array(y_gender)

        encoded_dict_dev = tokenizer(txts_dev, add_special_tokens=True, padding='max_length', max_length=max_length,
                                 truncation=True, return_attention_mask=True)

        input_ids_dev = encoded_dict_dev['input_ids']
        attention_masks_dev = encoded_dict_dev['attention_mask']

        input_ids_dev = np.array(input_ids_dev)
        attention_masks_dev = np.array(attention_masks_dev)
        y_dev = np.array(y_dev_prof)
        z_dev = np.array(y_dev_gender)

        encoded_dict_test = tokenizer(txts_test, add_special_tokens=True, padding='max_length', max_length=max_length,
                                     truncation=True, return_attention_mask=True)

        input_ids_test = encoded_dict_test['input_ids']
        attention_masks_test = encoded_dict_test['attention_mask']

        input_ids_test = np.array(input_ids_test)
        attention_masks_test = np.array(attention_masks_test)
        y_test = np.array(y_test_prof)
        z_test = np.array(y_test_gender)


        torch.save({"X_train": input_ids, "masks_train": attention_masks, "y_train": y, "z_train": z,
                    "X_dev": input_ids_dev, "masks_dev": attention_masks_dev, "y_dev": y_dev, "z_dev": z_dev,
                    "X_test": input_ids_test, "masks_test": attention_masks_test, "y_test": y_test, "z_test": z_test},
                   path)
        print(f"Saved to {path}")

dataset = args.dataset
if dataset == "moji":
    data_path = f"/your_path/data/moji/tokens_{model_version}_128.pt"
    if not os.path.exists(data_path):
        #tokenize_data(dataset, tokenizer, model_version, data_path)
        print('tokenize_data')
    data = torch.load(data_path)
    input_ids, attention_masks, y, z = data["X"], data["masks"], data["y"], data["z"]

    input_ids_train_dev, input_ids_test, y_train_dev, y_test, z_train_dev, z_test, attention_masks_train_dev, attention_masks_test = sklearn.model_selection.train_test_split(
        input_ids, y, z, attention_masks, random_state=args.run_id, stratify=y, test_size=0.25)

    input_ids_train, input_ids_dev, y_train, y_dev, z_train, z_dev, attention_masks_train, attention_masks_dev = sklearn.model_selection.train_test_split(
        input_ids_train_dev, y_train_dev, z_train_dev, attention_masks_train_dev, random_state=args.run_id,
        stratify=y_train_dev, test_size=0.133)



    input_ids_train, attention_masks_train, y_train, y_gender = balance_data(input_ids_train, attention_masks_train,
                                                                             y_train, z_train)
    input_ids_dev, attention_masks_dev, y_dev, y_dev_gender = balance_data(input_ids_dev, attention_masks_dev, y_dev,
                                                                           z_dev)
    input_ids_test, attention_masks_test, y_test, y_test_gender = balance_data(input_ids_test, attention_masks_test,
                                                                               y_test, z_test)

    # convert the data to tensors
    input_ids_train, attention_masks_train, y_train, y_gender = torch.tensor(input_ids_train).to(device), torch.tensor(
        attention_masks_train).to(device), torch.tensor(y_train).to(device), torch.tensor(y_gender).to(device)
    input_ids_dev, attention_masks_dev, y_dev, y_dev_gender = torch.tensor(input_ids_dev).to(device), torch.tensor(
        attention_masks_dev).to(device), torch.tensor(y_dev).to(device), torch.tensor(y_dev_gender).to(device)
    input_ids_test, attention_masks_test, y_test, y_test_gender = torch.tensor(input_ids_test).to(device), torch.tensor(
        attention_masks_test).to(device), torch.tensor(y_test).to(device), torch.tensor(y_test_gender).to(device)
else: #dataset == "bios"
    data_path = f"/home/shadi.isk/adv-sim/data/{dataset}/tokens_{model_version}_128.pt"
    if not os.path.exists(data_path):
        # make path
        tokenize_data(dataset, tokenizer, model_version, data_path)

    data = torch.load(data_path)
    input_ids_train, attention_masks_train, y_train, y_gender = data["X_train"], data["masks_train"], data["y_train"], data["z_train"]
    input_ids_dev, attention_masks_dev, y_dev, y_dev_gender = data["X_dev"], data["masks_dev"], data["y_dev"], data["z_dev"]
    input_ids_test, attention_masks_test, y_test, y_test_gender = data["X_test"], data["masks_test"], data["y_test"], data["z_test"]
    # convert the data to tensors
    input_ids_train, attention_masks_train, y_train, y_gender = torch.tensor(input_ids_train).to(device),\
                                                                torch.tensor(attention_masks_train).to(device), torch.tensor(y_train).to(device), torch.tensor(y_gender).to(device)
    input_ids_dev, attention_masks_dev, y_dev, y_dev_gender = torch.tensor(input_ids_dev).to(device), \
                                                                torch.tensor(attention_masks_dev).to(device), torch.tensor(y_dev).to(device), torch.tensor(y_dev_gender).to(device)
    input_ids_test, attention_masks_test, y_test, y_test_gender = torch.tensor(input_ids_test).to(device), \
                                                                torch.tensor(attention_masks_test).to(device), torch.tensor(y_test).to(device),torch.tensor(y_test_gender).to(device)



# print shapes
train_size = len(input_ids_train)
embedding_dim = 768
n_main_classes = 28 if dataset=="bios" else 2
n_att_classes = 2

batch_size = 10
num_attributes = 2


W = torch.nn.Linear(embedding_dim, n_main_classes)

#now load models from path
def load_models(model,W,path, lambda_threshold):
    model.load_state_dict(torch.load("{}/bert_L{}.pt".format(path, lambda_threshold)))
    W.load_state_dict(torch.load("{}/W_L{}.pt".format(path, lambda_threshold)))
    return model, W

model, W = load_models(model,W,path, lambda_threshold)
model.to(device)
W.to(device)
train_vectors = []
dev_vectors = []
test_vectors = []

with torch.no_grad():
    model.eval()
    batch_size = 128
    # extract train cls vectors
    for i in range(0, len(input_ids_train), batch_size):
        input_ids = input_ids_train[i:i + batch_size]
        attention_mask = attention_masks_train[i:i + batch_size]
        v = model(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :].detach().cpu().numpy()
        train_vectors.extend(v)
    train_vectors = np.array(train_vectors)
    #extract dev cls vectors
    for i in range(0, len(input_ids_dev), batch_size):
        input_ids = input_ids_dev[i:i + batch_size]
        attention_mask = attention_masks_dev[i:i + batch_size]
        v = model(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :].detach().cpu().numpy()
        dev_vectors.extend(v)
    dev_vectors = np.array(dev_vectors)
    #extract test cls vectors
    for i in range(0, len(input_ids_test), batch_size):
        input_ids = input_ids_test[i:i + batch_size]
        attention_mask = attention_masks_test[i:i + batch_size]
        v = model(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :].detach().cpu().numpy()
        test_vectors.extend(v)
    test_vectors = np.array(test_vectors)

print("train vectors shape: ", train_vectors.shape)

#save vectors
torch.save({"X_train": train_vectors, "y_train": y_train, "z_train": y_gender}, path + f"/train_vectors_{model_version}_128.pt")
torch.save({"X_dev": dev_vectors, "y_dev": y_dev, "z_dev": y_dev_gender}, path + f"/dev_vectors_{model_version}_128.pt")
torch.save({"X_test": test_vectors, "y_test": y_test, "z_test": y_test_gender}, path + f"/test_vectors_{model_version}_128.pt")


