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
from transformers import BertModel, BertTokenizer
import argparse
import time
import wandb
from collections import Counter

import torch.nn.functional as F
import math

DATA_PATH = "/home/shadi.isk/nullspace_projection-master/data/"


from transformers import BertModel, BertTokenizer, DebertaTokenizer, DebertaModel, AutoModel
from transformers.models.deberta_v2 import DebertaV2TokenizerFast

def rms(arr):
    return np.sqrt(np.mean(np.square(arr)))


def get_TPR(y_main, y_hat_main, y_protected):
    y_main, y_protected = y_main[:len(y_hat_main)], y_protected[:len(y_hat_main)]
    all_y = list(Counter(y_main).keys())
    protected_vals = defaultdict(dict)
    for label in all_y:
        for i in range(2):
            used_vals = (y_main == label) & (y_protected == i)
            y_label = y_main[used_vals]
            y_hat_label = y_hat_main[used_vals]
            protected_vals['y:{}'.format(label)]['p:{}'.format(i)] = (y_label == y_hat_label).mean()

    diffs = {}
    for k, v in protected_vals.items():
        vals = list(v.values())
        diffs[k] = vals[0] - vals[1]
    return protected_vals, diffs


def load_bios(group):
    path = DATA_PATH + "biasbios/"
    X = np.load(path + "{}_cls.npy".format(group))
    with open(path + "{}.pickle".format(group), "rb") as f:
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


def read_files():
    data_dir = DATA_PATH + 'twitter_dial/'
    with open(data_dir + 'pos_pos', 'r') as f:
        pos_pos = f.readlines()
        pos_pos = [map(int, sen.split(' ')) for sen in pos_pos]
    with open(data_dir + 'pos_neg', 'r') as f:
        pos_neg = f.readlines()
        pos_neg = [map(int, sen.split(' ')) for sen in pos_neg]
    with open(data_dir + 'neg_pos', 'r') as f:
        neg_pos = f.readlines()
        neg_pos = [map(int, sen.split(' ')) for sen in neg_pos]
    with open(data_dir + 'neg_neg', 'r') as f:
        neg_neg = f.readlines()
        neg_neg = [map(int, sen.split(' ')) for sen in neg_neg]
    return pos_pos, pos_neg, neg_pos, neg_neg


def load_moji():
    pos_pos, pos_neg, neg_pos, neg_neg = read_files()
    print(pos_pos[0])
    print(1 / 0)
    txts = []
    y, z = [], []
    for label in ['neg', 'pos']:
        for aa in ['neg', 'pos']:
            file_name = DATA_PATH + f'twitter_dial/{label}_{aa}'
            ctr = 0
            with open(file_name, 'rb') as f:
                for line in f.readlines():
                    txts.append(line.decode("utf-8", errors='ignore'))
                    if ctr % 100000 == 0:
                        print(line)
                        print(txts)
                    y.append(1 if label == 'pos' else 0)
                    z.append(1 if aa == 'pos' else 0)
                    ctr += 1

    # shuffle the three lists in the same order
    txts, y, z = sklearn.utils.shuffle(txts, y, z)
    return txts, np.array(y), np.array(z)


if True:
    parser = argparse.ArgumentParser(description="An argparse example")
    parser.add_argument('--model', type=str, default="bert")
    parser.add_argument('--adv', type=int, default=0, required=False)
    parser.add_argument('--run_id', type=int, default=0)
    parser.add_argument('--device', type=int, default=0, required=False)
    parser.add_argument('--opt', type=str, default="sgd", required=False)
    parser.add_argument('--iters', type=int, default=30000, required=False)
    parser.add_argument('--emb_mode', type=int, default=0, required=False)
    parser.add_argument('--set_emb', type=int, default=0, required=False)
    parser.add_argument('--lam_st', type=float, default=0, required=False)
    parser.add_argument('--lam_end', type=float, default=1, required=False)
    parser.add_argument('--gamma', type=float, default=5, required=False)
    parser.add_argument('--lr', type=float, default=0.01, required=False)
    parser.add_argument('--save', type=int, default=0, required=False)
    parser.add_argument('--learned_loss', type=int, default=1, required=False)
    parser.add_argument('--labaled_size', type=int, default=1000, required=False)
    parser.add_argument('--n_update', type=int, default=200, required=False)
    parser.add_argument('--n_evaluate', type=int, default=2000, required=False)
    parser.add_argument('--proj_name', type=str, default='', required=False)
    parser.add_argument('--dataset', type=str, default="moji", required=False)
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    adv = (args.adv == 1)

    save_models = (args.save == 1)
    set_emb = (args.set_emb >= 1)
    set_size = args.set_emb
    lambdaa = args.lam_st
    # TODO :
    alpha = 1

    g = args.gamma
    lambda_threshold = args.lam_end
    pre_defined = (args.emb_mode == 0)
    data_driven = (args.emb_mode == 1)
    data_learned = (args.emb_mode == 2)

    if args.emb_mode == -1:
        uni_sim = False
        lambda_threshold = 0
    else:
        uni_sim = True
    emb_mode_name = "reg" if not uni_sim else "pd" if pre_defined else "dd" if data_driven else "dl" if data_learned else "donno"

if args.model == "bert":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased")
    model_version = "bert-base-uncased"
elif args.model == "deberta":
    tokenizer = DebertaV2TokenizerFast.from_pretrained("microsoft/deberta-v3-large")
    bert = AutoModel.from_pretrained('microsoft/deberta-v3-base', output_attentions=False, output_hidden_states=False)
    model_version = "deberta-v3-base"
elif args.model == "deberta-base":
    tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
    bert = DebertaModel.from_pretrained('microsoft/deberta-base', output_attentions=False,
                                        output_hidden_states=False)
    model_version = "deberta-base"
elif args.model == "roberta":
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    bert = RobertaModel.from_pretrained('roberta-base')
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
    data_path = f"/home/shadi.isk/adv-sim/data/moji/tokens_{model_version}_128.pt"
    if not os.path.exists(data_path):
        # tokenize_data(dataset, tokenizer, model_version, data_path)
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
else:  # dataset == "bios"
    data_path = f"/home/shadi.isk/adv-sim/data/{dataset}/tokens_{model_version}_128.pt"
    if not os.path.exists(data_path):
        # make path
        tokenize_data(dataset, tokenizer, model_version, data_path)

    data = torch.load(data_path)
    input_ids_train, attention_masks_train, y_train, y_gender = data["X_train"], data["masks_train"], data["y_train"], \
                                                                data["z_train"]
    input_ids_dev, attention_masks_dev, y_dev, y_dev_gender = data["X_dev"], data["masks_dev"], data["y_dev"], data[
        "z_dev"]
    input_ids_test, attention_masks_test, y_test, y_test_gender = data["X_test"], data["masks_test"], data["y_test"], \
                                                                  data["z_test"]
    # convert the data to tensors
    input_ids_train, attention_masks_train, y_train, y_gender = torch.tensor(input_ids_train).to(device), \
                                                                torch.tensor(attention_masks_train).to(
                                                                    device), torch.tensor(y_train).to(
        device), torch.tensor(y_gender).to(device)
    input_ids_dev, attention_masks_dev, y_dev, y_dev_gender = torch.tensor(input_ids_dev).to(device), \
                                                              torch.tensor(attention_masks_dev).to(
                                                                  device), torch.tensor(y_dev).to(device), torch.tensor(
        y_dev_gender).to(device)
    input_ids_test, attention_masks_test, y_test, y_test_gender = torch.tensor(input_ids_test).to(device), \
                                                                  torch.tensor(attention_masks_test).to(
                                                                      device), torch.tensor(y_test).to(
        device), torch.tensor(y_test_gender).to(device)

# print sizes
print_data_size = True
if print_data_size:
    print("tr", len(input_ids_train))
    print("dev", len(input_ids_dev))
    print("test", len(input_ids_test))

# print shapes
train_size = len(input_ids_train)
embedding_dim = 768
n_main_classes = len(set(y_train.tolist()))
n_att_classes = 2

if args.adv:
    if args.adv == 1:
        adv_clf = torch.nn.Linear(embedding_dim, 2)
        adv_clf = torch.nn.Sequential(*[RevGrad(), adv_clf])
        adv_clf.to(device)
    if args.adv == 2:
        adv_clf = torch.nn.Sequential(
            *[RevGrad(), torch.nn.Linear(embedding_dim, 300), torch.nn.ReLU(), torch.nn.LayerNorm(300),
              torch.nn.Linear(300, 2)]).to(device)
    run_name = "Adv clf"

"""Initialize Parameters"""
W = torch.nn.Linear(embedding_dim, n_main_classes)
loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
loss_fn_kl = torch.nn.KLDivLoss(reduction="batchmean")
# loss_fn_kl = torch.nn.CrossEntropyLoss()
W.to(device)
bert.to(device)
bert.train()

"""Initialize Gender Embeddings"""
male_cls_all = torch.FloatTensor([]).to(device)
female_cls_all = torch.FloatTensor([]).to(device)
male_labeled_idx = None
female_labeled_idx = None


def eval_dev(bert, W, input_ids_dev, attention_mask_dev, y_dev, device, adv=None, y_dev_gender=None, uni_sim=False):
    loss_vals = []
    bert.eval()
    batch_size = 64
    accs = []
    if adv is not None:
        accs_adv = []
        adv_loss = 0
    if uni_sim:
        sim_loss = 0
        accs_sim = []
        num_attributes = 2
        uniform_sim = torch.zeros(batch_size, num_attributes) + 0.5

    with torch.no_grad():
        print("Evaluating...")
        pbar = tqdm.tqdm(range(len(input_ids_dev)), ascii=True)
        y_preds = torch.LongTensor([]).to(device)
        for i in range(0, len(input_ids_dev) - batch_size, batch_size):
            batch_input_ids = input_ids_dev[i: i + batch_size]
            batch_attention_mask = attention_mask_dev[i: i + batch_size]
            batch_y = y_dev[i: i + batch_size]

            # input_ids, token_type_ids, attention_mask = batch_encoding["input_ids"], batch_encoding["token_type_ids"], \
            #                                             batch_encoding["attention_mask"]
            input_ids = batch_input_ids
            # token_type_ids = torch.tensor(token_type_ids).to(device)
            attention_mask = batch_attention_mask
            """POOLER OUTPUT"""
            H = bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
            logits = W(H)
            loss = loss_fn(logits, torch.tensor(batch_y).to(device))
            loss_vals.append(loss.detach().cpu().numpy())

            preds = logits.argmax(dim=1)
            acc = (preds == torch.tensor(batch_y).to(device)).float().mean().detach().cpu().numpy()
            y_preds = torch.cat([y_preds, preds]).to(device)
            accs.append(acc)
            if False:
              if adv:
                  logits_adv = adv(H)
                  batch_y_gender = y_dev_gender[i: i + batch_size]
                  preds_adv = logits_adv.argmax(dim=1)
                  adv_loss += loss_fn(logits_adv, torch.tensor(batch_y_gender).to(device))
                  acc = (preds_adv == torch.tensor(batch_y_gender).to(device)).float().mean().detach().cpu().numpy()
                  accs_adv.append(acc)
              if uni_sim:
                  batch_y_gender = y_dev_gender[i: i + batch_size]
                  for k in range(set_size):
                      male_sim_sc = (H @ male_cls_all[k].T).squeeze().to(
                          device)  # torch.cosine_similarity(H, male_cls, dim=1)
                      female_sim_sc = (H @ female_cls_all[k].T).squeeze().to(
                          device)  # torch.cosine_similarity(H, female_cls, dim=1)
                      sim_scores = torch.stack([male_sim_sc, female_sim_sc], dim=1)
                      input_sim = torch.log_softmax(sim_scores, dim=1)
                      sim_loss += loss_fn_kl(input_sim.to(device), uniform_sim.to(device)).to(device)
                  preds_sim = torch.argmax(input_sim, dim=1)
                  acc = (preds_sim == torch.tensor(batch_y_gender).to(device)).float().mean().detach().cpu().numpy()
                  accs_sim.append(acc)
            pbar.update(batch_size)
    bert.train()
    y_dev, y_dev_gender = y_dev[:len(y_preds)], y_dev_gender[:len(y_preds)]

    dev_gap = rms(list(get_TPR(y_dev, y_preds.detach().cpu().numpy(), y_dev_gender)[1].values()))
    return_dict = {"loss": np.mean(loss_vals), "acc": np.mean(accs) * 100, "tpr-gap": dev_gap * 100}
    if False and adv:
        return_dict["adv-acc"] = np.mean(accs_adv) * 100
        return_dict["adv-loss"] = (adv_loss * batch_size / len(input_ids_dev))
    if False and uni_sim:
        sim_loss = lambdaa * (sim_loss / set_size)
        return_dict["sim-loss"] = (sim_loss * batch_size / len(input_ids_dev))
        return_dict["sim-acc"] = np.mean(accs_sim) * 100
    return return_dict, y_dev, y_dev_gender, y_preds


#now load models from path
def load_models(model,W,path, lambda_threshold):
    model.load_state_dict(torch.load("{}/bert_L{}.pt".format(path, lambda_threshold)))
    W.load_state_dict(torch.load("{}/W_L{}.pt".format(path, lambda_threshold)))
    return model, W

path = f"models/{args.dataset}/{args.model}/{emb_mode_name}/{args.run_id}"
bert, W = load_models(bert, W, path, lambda_threshold)
print('here')
return_dict, y_test, y_test_gender, y_test_preds = eval_dev(bert, W, input_ids_test, attention_masks_test, y_test.cpu().numpy(), device,
                       adv=adv_clf if adv else None,
                       y_dev_gender=y_test_gender.cpu().numpy(), uni_sim=uni_sim)

torch.save(y_test, f"{path}/y.pt")
torch.save(y_test_gender, f"{path}/z.pt")
torch.save(y_test_preds, f"{path}/y_pred.pt")
print('saved')

test_acc, test_gap =  return_dict["acc"], return_dict["tpr-gap"]

from allennlp.fairness import Independence, Separation, Sufficiency
def dictionary_torch_to_number(d: dict):
    for k, v in d.items():
        if isinstance(v, dict):
            dictionary_torch_to_number(v)
        else:
            d[k] = v.item()
    sum = 0
    for k, v in d.items():
        sum += v
independence = Independence(n_main_classes, n_att_classes)
independence(y_test_preds, y_test_gender)
independence_score = independence.get_metric()
independence_sum = dictionary_torch_to_number(independence_score)

print("independence_sum = ", independence_sum)
print("test_accuracy = ", test_acc)
print("test_tpr-gap  = ", test_gap)


