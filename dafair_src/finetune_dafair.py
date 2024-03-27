import sys
import os

from sklearn.linear_model import SGDClassifier, LinearRegression, Lasso, Ridge
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
import seaborn as sn
import random
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
from sklearn.manifold import TSNE
import tqdm
from transformers import RobertaTokenizer, RobertaModel, DebertaModel, BertModel, DebertaTokenizer
from transformers import AutoModel
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
from transformers import BertModel, BertTokenizer, DebertaTokenizer, DebertaModel
from transformers.models.deberta_v2 import DebertaV2TokenizerFast
import argparse
import time
import wandb

import torch.nn.functional as F
import math
DATA_PATH = "/your_path/data/"
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

    adv = (args.adv == 1)


    save_models = (args.save == 1)
    set_emb = (args.set_emb >= 1)
    set_size = args.set_emb
    lambdaa = args.lam_st

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

    if save_models:
        if not os.path.exists("models"):
            os.makedirs("models")
        if not os.path.exists(f"models/{args.dataset}"):
            os.makedirs(f"models/{args.dataset}")
        if not os.path.exists(f"models/{args.dataset}/{args.model}"):
            os.makedirs(f"models/{args.dataset}/{args.model}")
        if not os.path.exists(f"models/{args.dataset}/{args.model}/{emb_mode_name}"):
            os.makedirs(f"models/{args.dataset}/{args.model}/{emb_mode_name}")
        if not os.path.exists(f"models/{args.dataset}/{args.model}/{emb_mode_name}/N_{args.set_emb}"):
            os.makedirs(f"models/{args.dataset}/{args.model}/{emb_mode_name}/N_{args.set_emb}")
        if not os.path.exists(f"models/{args.dataset}/{args.model}/{emb_mode_name}/N_{args.set_emb}/{args.run_id}"):
            os.makedirs(f"models/{args.dataset}/{args.model}/{emb_mode_name}/N_{args.set_emb}/{args.run_id}")
        path = f"models/{args.dataset}/{args.model}/{emb_mode_name}/N_{args.set_emb}/{args.run_id}"
# Random seed
random.seed(args.run_id)
np.random.seed(args.run_id)
torch.manual_seed(args.run_id)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# K
n_partition = set_size

# Initialize base model
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
    bert = DebertaModel.from_pretrained('microsoft/deberta-base', output_attentions=False, output_hidden_states=False)
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
    data_path = f"/your_path/data/{dataset}/tokens_{model_version}_128.pt"
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



#print data sizes
print_data_size = True
if print_data_size:
    print("tr",len(input_ids_train))
    print("dev",len(input_ids_dev))
    print("test",len(input_ids_test))


#print shapes
train_size = len(input_ids_train)
embedding_dim = 768
n_main_classes = len(torch.unique(y_train))
n_update_gender_emb = args.n_update
num_attributes = 2 # we used debiasing for binary attributes (male/female or AAE,SAE)
batch_size = 10
epochs = 1
n_iters = (train_size // batch_size) * epochs

# we did use adv
if adv:
    if args.adv == 1:
        adv_clf = torch.nn.Linear(embedding_dim, 2)
        adv_clf = torch.nn.Sequential(*[RevGrad(), adv_clf])
        adv_clf.to(device)
    if args.adv == 2:
        adv_clf = torch.nn.Sequential(*[RevGrad(), torch.nn.Linear(embedding_dim, 300), torch.nn.ReLU(), torch.nn.LayerNorm(300),
                                        torch.nn.Linear(300, 2)]).to(device)
    run_name = "Adv clf"


"""Initialize Parameters"""
W = torch.nn.Linear(embedding_dim, n_main_classes)
loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
loss_fn_kl = torch.nn.KLDivLoss(reduction="batchmean")
#loss_fn_kl = torch.nn.CrossEntropyLoss()
W.to(device)
bert.to(device)
bert.train()

"""Initialize Gender Embeddings"""
male_cls_all = torch.FloatTensor([]).to(device)
female_cls_all = torch.FloatTensor([]).to(device)
male_labeled_idx = None
female_labeled_idx = None

male_prototypical_texts = [
    "This is a biography about a male.",
    "A man who excelled in his field.",
    "He is known for his achievements in various industries.",
    "A prominent male figure in history.",
    "His career and accomplishments are well-regarded.",
    "This biography focuses on the life of a distinguished man.",
    "An influential male individual.",
    "He made significant contributions to his profession.",
    "This is a story about a man who shaped his industry.",
    "His impact on his field is noteworthy."
]
female_prototypical_texts = [
    "This is a biography about a female.",
    "A woman who excelled in her field.",
    "She is known for her achievements in various industries.",
    "A prominent female figure in history.",
    "Her career and accomplishments are well-regarded.",
    "This biography focuses on the life of a distinguished woman.",
    "An influential female individual.",
    "She made significant contributions to her profession.",
    "This is a story about a woman who shaped her industry.",
    "Her impact on her field is noteworthy."
]
white_prototypical_texts = [
    "This tweet reflects a [sentiment] from a white writer.",
    "A tweet expressing a [sentiment] moment by a white individual.",
    "A [sentiment] viewpoint shared by a writer using white sociolect.",
    "This post, written in standard English, conveys [sentiment] from a white perspective.",
    "A message filled with [sentiment] from a white communicator.",
    "A white person shares their [sentiment] thoughts in this tweet.",
    "This is an example of a tweet with [sentiment] in white sociolect.",
    "A tweet written by a white speaker that conveys [sentiment].",
    "This post by a white individual radiates [sentiment] and [sentiment].",
    "A [sentiment] perspective presented by a writer using white sociolect."
]

black_prototypical_texts = [
    "This tweet reflects a [sentiment] sentiment from a black writer.",
    "A tweet expressing a [sentiment] moment by a black individual.",
    "A [sentiment] viewpoint shared by a writer using African American English.",
    "This post, written in AAE, conveys [sentiment] from a black perspective.",
    "A message filled with [sentiment] from a black communicator.",
    "A black person shares their [sentiment] thoughts in this tweet.",
    "This is an example of a tweet with [sentiment] sentiment in AAE.",
    "A tweet written by a black speaker that conveys [sentiment].",
    "This post by a black individual radiates [sentiment] and [sentiment].",
    "A [sentiment] perspective presented by a writer using African American English."
]

text_length = range(len(black_prototypical_texts))
index_random = random.sample(text_length, set_size)
random_male_texts = [male_prototypical_texts[index] for index in index_random]
random_female_texts = [female_prototypical_texts[index] for index in index_random]
random_white_texts = [white_prototypical_texts[index] for index in index_random]
random_black_texts = [black_prototypical_texts[index] for index in index_random]


# Choose DaFAIR (pre_defined) or SemiDAFair (data_driven). (data_learned is similar to Adversarial learning)
if uni_sim:
    if pre_defined:
        if dataset == "moji":
            male_text_all = random_black_texts
            female_text_all = random_white_texts
            
        elif dataset == "bios":
            male_text_all = random_male_texts
            female_text_all = random_female_texts
             
        else:
            print('dataset not supported')
            # terminate
            raise NotImplementedError

        run_name = f"Pre-Defined {set_size} pair"
    if data_driven:
        m = args.labaled_size
        perm = torch.randperm(train_size)
        labeled_idx = perm[:m+100]
        labeled_idx_partitions = torch.split(labeled_idx, n_partition)
        male_labeled_idx = labeled_idx[y_gender[labeled_idx] == 0][:m//2]
        female_labeled_idx = labeled_idx[y_gender[labeled_idx] == 1][:m//2]
        run_name = f"Data-Driven E{m}_N{set_size}"
    if data_learned:
        # for random 1000 samples of X_train we have labels
        m = args.labaled_size
        perm = torch.randperm(train_size)
        labeled_idx = perm[:m+100]
        male_labeled_idx = labeled_idx[y_gender[labeled_idx] == 0][:m//2]
        female_labeled_idx = labeled_idx[y_gender[labeled_idx] == 1][:m//2]
        male_cls_all = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(embedding_dim, requires_grad=True).to(device)) for _ in range(set_size)])
        female_cls_all = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(embedding_dim, requires_grad=True).to(device)) for _ in range(set_size)])

        emb_optimizer = torch.optim.Adam([
            {'params': male_cls_all.parameters()},
            {'params': female_cls_all.parameters()}  # Pass the list of parameters here
        ], lr=args.lr)
        run_name = f"Data-Learned E{m}_N{set_size}"
        if args.learned_loss == 2:
            run_name = f"Data-Learned L2_E{m}_N{set_size}"
        if args.learned_loss == 11:
                run_name = f"Data-Learned L1B_E{m}_N{set_size}"
        if args.learned_loss == 22:
                run_name = f"Data-Learned L2B_E{m}_N{set_size}"
else:
    run_name = "No Sim Adv"

# optimizer
if args.opt == "sgd":
    if not adv:
        lr, momentum, decay = 0.5 * 1e-4, 0.9, 1e-6
        optimizer = torch.optim.SGD(list(bert.parameters()) + list(W.parameters(())), lr=lr, momentum=momentum,
                                    weight_decay=decay)
    else:
        lr, momentum, decay = 0.5 * 1e-4, 0.8, 1e-6
        optimizer = torch.optim.SGD(list(bert.parameters()) + list(W.parameters()) + list(adv_clf.parameters()), lr=lr,
                                    momentum=momentum, weight_decay=decay)
else:
    lr, momentum, decay = None, None, None
    if not adv:
        optimizer = torch.optim.Adam(list(bert.parameters()) + list(W.parameters()), lr=5e-5)
    else:
        optimizer = torch.optim.Adam(list(bert.parameters()) + list(W.parameters()) + list(adv_clf.parameters()),
                                     lr=5e-5)


n_to_evaluate = args.n_evaluate
loss_vals,ce_loss_vals,kl_loss_vals = [],[],[]
best_score = 10000
best_model = None

pbar = tqdm.tqdm(range(n_iters), ascii=True)
d, d2 = args.__dict__, {"ce_lr": lr, "momentum": momentum, "decay": decay}
new_d = {**d, **d2}
run_name += f"_L{args.lam_end}_"
run_name = f"ChatGPT_random_{emb_mode_name}_N{set_size}_L{args.lam_end}_{args.model}"
run = wandb.init(project=f"DAFair {args.proj_name}", config=new_d,name=run_name,
                 settings=wandb.Settings(start_method='fork'),tags=["PD-Pt" if args.emb_mode == 0 and (not set_emb) else "PD-Set" if set_emb else "DD" if data_driven else "DL"])

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
            H = bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0,:]
            logits = W(H)
            loss = loss_fn(logits, torch.tensor(batch_y).to(device))
            loss_vals.append(loss.detach().cpu().numpy())

            preds = logits.argmax(dim=1)
            acc = (preds == torch.tensor(batch_y).to(device)).float().mean().detach().cpu().numpy()
            y_preds = torch.cat([y_preds,preds]).to(device)
            accs.append(acc)

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
                preds_sim = torch.argmax(input_sim,dim=1)
                acc = (preds_sim == torch.tensor(batch_y_gender).to(device)).float().mean().detach().cpu().numpy()
                accs_sim.append(acc)
            pbar.update(batch_size)
    bert.train()
    y_dev, y_dev_gender = y_dev[:len(y_preds)], y_dev_gender[:len(y_preds)]

    dev_gap = rms(list(get_TPR(y_dev, y_preds.detach().cpu().numpy(), y_dev_gender)[1].values()))
    return_dict = {"loss": np.mean(loss_vals), "acc": np.mean(accs)*100, "tpr-gap": dev_gap*100}
    if adv:
        return_dict["adv-acc"] = np.mean(accs_adv)*100
        return_dict["adv-loss"] = ( adv_loss*batch_size / len(input_ids_dev) )
    if uni_sim:
        sim_loss = lambdaa * (sim_loss / set_size)
        return_dict["sim-loss"] = ( sim_loss*batch_size / len(input_ids_dev) )
        return_dict["sim-acc"] = np.mean(accs_sim)*100
    return return_dict


def pd_gender_emb(bert,device):
    bert.eval()
    male_cls_all = torch.FloatTensor([]).to(device)
    female_cls_all = torch.FloatTensor([]).to(device)
    for k in range(set_size):
        # Tokenize male text
        male_text = male_text_all[k]
        tokens = tokenizer.tokenize(male_text)
        input_ids_k = tokenizer.convert_tokens_to_ids(tokens)
        male_input_tensor = torch.tensor([input_ids_k]).to(device)
        outputs = bert(male_input_tensor)
        male_cls_k = outputs.last_hidden_state[:,0,:].detach().to(device)
        male_cls_all = torch.cat([male_cls_all,male_cls_k])
        #Tokenize female text
        female_text = female_text_all[k]
        tokens = tokenizer.tokenize(female_text)
        input_ids_k = tokenizer.convert_tokens_to_ids(tokens)
        female_input_tensor = torch.tensor([input_ids_k]).to(device)
        outputs = bert(female_input_tensor)
        female_cls_k = outputs.last_hidden_state[:,0,:].detach().to(device)
        female_cls_all = torch.cat([female_cls_all, female_cls_k])

        similarity_score = torch.cosine_similarity(male_cls_k, female_cls_k, dim=1)
        print("score between the two representations {}".format(similarity_score))
    bert.train()
    return male_cls_all,female_cls_all


def dd_gender_emb(bert, input_ids_, attention_mask_, male_labeled_idx, female_labeled_idx, device):
    bert.eval()
    batch_size = 128 if len(male_labeled_idx) > 500 else 32 if len(male_labeled_idx) > 100 else 2
    X_male = torch.FloatTensor([]).to(device)
    X_female = torch.FloatTensor([]).to(device)
    with torch.no_grad():
        for i in range(0, len(male_labeled_idx) - batch_size, batch_size):
            batch_input_ids = input_ids_[male_labeled_idx[i: i + batch_size]].to(device)
            batch_attention_mask = attention_mask_[male_labeled_idx[i: i + batch_size]].to(device)
            """POOLER OUTPUT"""
            H = bert(input_ids=batch_input_ids, attention_mask=batch_attention_mask).last_hidden_state[:, 0, :].to(device)
            X_male = torch.cat([X_male, H])
        for i in range(0, len(female_labeled_idx) - batch_size, batch_size):
            batch_input_ids = input_ids_[female_labeled_idx[i: i + batch_size]].to(device)
            batch_attention_mask = attention_mask_[female_labeled_idx[i: i + batch_size]].to(device)
            """POOLER OUTPUT"""
            H = bert(input_ids=batch_input_ids, attention_mask=batch_attention_mask).last_hidden_state[:, 0, :].to(device)
            X_female = torch.cat([X_female, H])

        X_male_partitions = torch.stack(torch.chunk(X_male, n_partition), dim=1).to(device)
        X_female_partitions = torch.stack(torch.chunk(X_female, n_partition), dim=1).to(device)
        male_cls_all = torch.mean(X_male_partitions, dim=0)
        female_cls_all = torch.mean(X_female_partitions, dim=0)
    bert.train()
    return male_cls_all, female_cls_all

def get_gender_emb(bert,input_ids, attention_mask,male_labeled_idx,female_labeled_idx,male_cls_all, female_cls_all,device):
    if pre_defined:
        male_cls_all, female_cls_all = pd_gender_emb(bert, device)
    if data_learned:
        male_cls_all, female_cls_all = ft_gender_emb(bert, input_ids, attention_mask, male_labeled_idx, female_labeled_idx,
                                                     male_cls_all, female_cls_all,device)
    if data_driven:
        male_cls_all, female_cls_all = dd_gender_emb(bert,input_ids, attention_mask,male_labeled_idx,female_labeled_idx,device)
    return male_cls_all, female_cls_all

def ft_gender_emb(bert, input_ids, attention_mask, male_labeled_idx, female_labeled_idx, male_cls_all, female_cls_all, device):
    bert.eval()
    batch_size = 32 if len(male_labeled_idx) > 100 else 2
    loss = 0
    male_input_ids = input_ids[male_labeled_idx]
    female_input_ids = input_ids[female_labeled_idx]
    male_attention_mask = attention_mask[male_labeled_idx]
    female_attention_mask = attention_mask[female_labeled_idx]

    male_H_all = torch.FloatTensor([]).to(device)
    female_H_all = torch.FloatTensor([]).to(device)

    for i in range(0,len(male_labeled_idx)-batch_size,batch_size):
        batch_input_ids = male_input_ids[i: i + batch_size]
        batch_attention_mask = male_attention_mask[i: i + batch_size]

        input_ids = batch_input_ids.to(device)
        attention_mask = batch_attention_mask.to(device)

        H = bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :].detach().to(device)

        male_H_all = torch.cat([male_H_all, H])

    #for i in range(0,len(female_labeled_idx)-batch_size,batch_size):
        batch_input_ids = female_input_ids[i: i + batch_size]
        batch_attention_mask = female_attention_mask[i: i + batch_size]

        input_ids = batch_input_ids.to(device)
        attention_mask = batch_attention_mask.to(device)

        H = bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :].detach().to(device)

        female_H_all = torch.cat([female_H_all, H])
        if args.learned_loss > 10:
            for k in range(set_size):
                male_sim = torch.cosine_similarity(male_H_all, male_cls_all[k])
                female_sim = torch.cosine_similarity(female_H_all, female_cls_all[k])
                male_female_sim = torch.cosine_similarity(female_H_all, male_cls_all[k])
                female_male_sim = torch.cosine_similarity(male_H_all, female_cls_all[k])
                # compute loss: maximize similarity with respective class embedding
                # can add - male-female & - fem-male
                if args.learned_loss == 11:
                    loss += (-1 * (male_sim.mean() + female_sim.mean()))/set_size  # - male_female_sim.mean() - female_male_sim.mean())
                if args.learned_loss == 22:
                    loss += (-1 * (male_sim.mean() + female_sim.mean() - male_female_sim.mean() - female_male_sim.mean()))/set_size
    if args.learned_loss < 10:
        for k in range(set_size):
            male_sim = torch.cosine_similarity(male_H_all, male_cls_all[k])
            female_sim = torch.cosine_similarity(female_H_all, female_cls_all[k])
            male_female_sim = torch.cosine_similarity(female_H_all, male_cls_all[k])
            female_male_sim = torch.cosine_similarity(male_H_all, female_cls_all[k])
            # compute loss: maximize similarity with respective class embedding
            # can add - male-female & - fem-male
            if args.learned_loss == 1:
                loss += -1 * (male_sim.mean() + female_sim.mean())# - male_female_sim.mean() - female_male_sim.mean())
            if args.learned_loss == 2:
                loss += -1 * (male_sim.mean() + female_sim.mean() - male_female_sim.mean() - female_male_sim.mean())

        loss = loss/set_size
    # backward pass and optimization step
    emb_optimizer.zero_grad()
    loss.backward()
    emb_optimizer.step()

    bert.train()
    return male_cls_all, female_cls_all


if uni_sim:
    uniform_sim = torch.zeros(batch_size, num_attributes) + 0.5 # repeat to match input_sim size
    #uniform_sim = F.softmax(uniform_batch.to(device), dim=1)
dev_ce_loss, dev_sim_loss, dev_gap, dev_acc = 0, 0, 0, 0
ce_loss, kl_loss, adv_loss, dev_sim_loss, dev_adv_acc, dev_sim_acc = 0, 0, 0, 0, -1,-1
test_acc, test_gap = 0, 0
best_score_tpr, best_score_acc = 0, 0
male_cls_all, female_cls_all = get_gender_emb(bert, input_ids_train, attention_masks_train, male_labeled_idx, female_labeled_idx,male_cls_all, female_cls_all, device)
male_cls_all.to(device)
female_cls_all.to(device)

evaluate = False

# Main Loop of training!
for i in pbar:

    optimizer.zero_grad()

    idx = np.arange(train_size)
    random.shuffle(idx)
    batch_idx = idx[:batch_size]
    #get batches
    input_ids, attention_mask, batch_sentiment, batch_att = input_ids_train[batch_idx], attention_masks_train[batch_idx], y_train[batch_idx], y_gender[batch_idx]
    #batch_texts, batch_prof = [txts_train[i] for i in batch_idx], y_train[batch_idx]

    if evaluate:
        if i==0 or (i + 1) % (n_to_evaluate) == 0:
            return_dict = eval_dev(bert, W, input_ids_dev, attention_masks_dev, y_dev.cpu().numpy(), device, adv=adv_clf if adv else None,
                                   y_dev_gender=y_dev_gender.cpu().numpy(), uni_sim=uni_sim)
            dev_ce_loss, dev_acc, dev_gap = return_dict["loss"], return_dict["acc"], return_dict["tpr-gap"]
            if adv:
                dev_adv_acc = return_dict["adv-acc"]
                dev_adv_loss = return_dict["adv-loss"]
            if uni_sim:
                dev_sim_loss = return_dict["sim-loss"]
                dev_sim_acc = return_dict["sim-acc"]
            train_loss = np.mean(loss_vals)
            train_ce_loss = np.mean(ce_loss_vals)
            train_kl_loss = np.mean(kl_loss_vals)
            # ""Careful we take best model regarding dst performance only ""
            if dev_ce_loss + dev_sim_loss < best_score:
                best_score = (dev_ce_loss.copy() + dev_sim_loss)
                best_score_tpr = dev_gap
                best_score_acc = dev_acc
            return_dict = eval_dev(bert, W, input_ids_test, attention_masks_test, y_test.cpu().numpy(), device, adv=adv_clf if adv else None,
                                   y_dev_gender=y_test_gender.cpu().numpy(), uni_sim=uni_sim)
            test_ce_loss, test_acc, test_gap = return_dict["loss"], return_dict["acc"], return_dict["tpr-gap"]
            log_dict = {"Lambda": lambdaa, "train_loss": train_loss, "train_ce_loss": train_ce_loss,
                        "train_kl_loss": train_kl_loss, "adv_loss": adv_loss,
                        "dev_ce_loss": dev_ce_loss, "lowest_loss": best_score * 100, "dev_acc": dev_acc,
                        "dev_tpr-gap": dev_gap,
                        "dev_adv_acc": dev_adv_acc, "dev_sim-loss": dev_sim_loss, "dev_sim-acc": dev_sim_acc,
                        "test_acc": test_acc, "test_tpr-gap": test_gap}#, "best_tpr-gap": best_score_tpr,
                        #"best_acc": best_score_acc}
            wandb.log(log_dict)

    if (i+1) % n_update_gender_emb == 0:
        loss_vals,ce_loss_vals,kl_loss_vals = [],[],[]
        male_cls_all, female_cls_all = get_gender_emb(bert, input_ids_train, attention_masks_train, male_labeled_idx, female_labeled_idx, male_cls_all, female_cls_all, device)
        male_cls_all.to(device)
        female_cls_all.to(device)


    H = bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0,:]#["pooler_output"]
    logits = W(H)
    ce_loss = loss_fn(logits, batch_sentiment)
    loss = ce_loss
    if adv:
        adv_loss = loss_fn(adv_clf(H), batch_att)
        loss = ce_loss + adv_loss
    if uni_sim:
        kl_loss = 0
        for k in range(set_size):
            male_sim_sc = (H @ male_cls_all[k].T).squeeze().to(device)  # torch.cosine_similarity(H, male_cls, dim=1)
            female_sim_sc = (H @ female_cls_all[k].T).squeeze().to(
                device)  # torch.cosine_similarity(H, female_cls, dim=1)
            sim_scores = torch.stack([male_sim_sc, female_sim_sc], dim=1)
            input_sim = torch.log_softmax(sim_scores, dim=1)
            kl_loss += loss_fn_kl(input_sim.to(device), uniform_sim.to(device)).to(device)
        kl_loss = lambdaa * (kl_loss / set_size)
        loss = ce_loss + kl_loss

    loss.backward()
    optimizer.step()

    loss_vals.append(loss.detach().cpu().numpy().item())
    ce_loss_vals.append(ce_loss.detach().cpu().numpy().item())
    if uni_sim:
        kl_loss_vals.append(kl_loss.detach().cpu().numpy().item())
    lambdaa = (2/(1+math.exp(-g*i/len(pbar)))-1)*lambda_threshold + args.lam_st


# dev evaluation
return_dict = eval_dev(bert, W, input_ids_dev, attention_masks_dev, y_dev.cpu().numpy(), device,
                       adv=adv_clf if adv else None,
                       y_dev_gender=y_dev_gender.cpu().numpy(), uni_sim=uni_sim)
dev_ce_loss, dev_acc, dev_gap = return_dict["loss"], return_dict["acc"], return_dict["tpr-gap"]
wandb.run.summary["dev_acc_after"] = dev_acc
wandb.run.summary["dev_gap_after"] = dev_gap


# test evaluation
return_dict = eval_dev(bert, W, input_ids_test, attention_masks_test, y_test.cpu().numpy(), device,
                       adv=adv_clf if adv else None,
                       y_dev_gender=y_test_gender.cpu().numpy(), uni_sim=uni_sim)
test_ce_loss, test_acc, test_gap = return_dict["loss"], return_dict["acc"], return_dict["tpr-gap"]
wandb.run.summary["acc_after"] = test_acc
wandb.run.summary["test_gap_after"] = test_gap
if save_models:
    torch.save(W.state_dict(), "{}/W_L{}.pt".format(path, lambda_threshold))
    torch.save(bert.state_dict(), "{}/bert_L{}.pt".format(path, lambda_threshold))
    if data_learned:
        torch.save(male_cls_all, "{}/male_cls.pt".format(path))
        torch.save(female_cls_all, "{}/female_cls.pt".format(path))
    if adv:
        torch.save(adv_clf.state_dict(), "{}/adv_L{}.pt".format(path, lambda_threshold))



run.finish()



