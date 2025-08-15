# %%
import os
import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import scipy.sparse as sp
import seaborn as sns
from datasets import load_from_disk
import torch.nn as nn, torch.nn.functional as F
import torch, json
from transformers import BertTokenizer, BertModel
from utils import BertModel as MedBertModel
from utils import LangCellDataCollatorForCellClassification as DataCollatorForCellClassification
from utils import LangCellTranscriptomeTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
import subprocess
from sklearn.metrics import accuracy_score, f1_score


#GPU_NUMBER = [args.device]
#os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])

nshot = 2
epochs = 2
seed = 42
train_batchsize = 4 
test_batchsize = 64
eval_num = 500
n_top_genes = 200

class Pooler(nn.Module):
    def __init__(self, config, pretrained_proj, proj_dim):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, proj_dim)
        self.proj.load_state_dict(torch.load(pretrained_proj))
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled_output = hidden_states[:, 0]
        pooled_output = F.normalize(self.proj(pooled_output), dim=-1)
        return pooled_output

def load_model():
    model = BertModel.from_pretrained('../data/ckpt/cell_bert')
    model.pooler = Pooler(model.config, pretrained_proj='../data/ckpt/cell_proj.bin', proj_dim=256)
    proj = model.pooler.proj
    # model = model.module
    model = model.to("cuda")

    text_pretrained_model = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    tokenizer = BertTokenizer.from_pretrained(text_pretrained_model)
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    text_encoder = MedBertModel.from_pretrained('../data/ckpt/text_bert', add_pooling_layer=True)
    text_encoder.pooler = Pooler(text_encoder.config, pretrained_proj='../data/ckpt/text_proj.bin', proj_dim=256)
    text_encoder = text_encoder.to("cuda")

    ctm_head = nn.Linear(text_encoder.config.hidden_size, 2)
    ctm_head.load_state_dict(torch.load('../data/ckpt/ctm_head.bin'))
    ctm_head = ctm_head.to("cuda")
    return model, tokenizer, text_encoder, ctm_head

def text_encode(text):
    text = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt').to('cuda')
    text = text_encoder(**text).pooler_output
    # text = F.normalize(model.text_projector(text))
    return text

def cell_encode(cell_input_ids, cell_atts):
    cell = model(cell_input_ids.to("cuda"), cell_atts.to("cuda"))
    cell_last_h = cell.last_hidden_state
    cell_pooler = cell.pooler_output
    return cell_last_h, cell_pooler

def ctm(text, cell_emb, cell_atts):
    # n texts, n cells -> n scores
    text = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt').to('cuda')
    output = text_encoder(**text,
                encoder_hidden_states = cell_emb.to("cuda"),
                encoder_attention_mask = cell_atts.to("cuda"),
                return_dict = True,
                mode = 'multimodal',
                )
    logits = ctm_head(output.last_hidden_state[:, 0, :])
    logits = F.softmax(logits, dim=-1)[..., 1] # [n]
    return logits

def calculate_mean_max_min(adata, feature_name):
    mean = np.mean(adata.obs[feature_name])
    median = np.median(adata.obs[feature_name])
    max_val = np.max(adata.obs[feature_name])
    min_val = np.min(adata.obs[feature_name])
    return mean, median, max_val, min_val

def load_dataset(tokenizer, text_encoder):
    ## Required features in tokenised dataset: ['input_ids', 'n_counts', 'batch', 'labels', 'n_genes', 'filter_pass', 'label']
    data_path = "../data/tablula_sapiens/Kidney_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad"
    #data_path = "../data/data_zeroshot/pbmc10k.dataset"
    #tokenized_dataset = load_from_disk(data_path)
    
    dataset_sub = sc.read(data_path)
    #dataset_sub = dataset_sub.shuffle(seed) #.select(range(300))
    print(dataset_sub)

    name_without_ext = os.path.splitext(os.path.basename(data_path))[0]

    sc.pp.highly_variable_genes(dataset_sub, n_top_genes=n_top_genes)
    dataset_sub = dataset_sub[:, dataset_sub.var.highly_variable]

    #print(dataset_sub)

    print(f"n_genes_by_counts (mean, med, max, min): {calculate_mean_max_min(dataset_sub, 'n_genes_by_counts')}")
    print(f"total_counts (mean, med, max, min): {calculate_mean_max_min(dataset_sub, 'total_counts')}")
    print(f"pct_counts_mt (mean, med, max, min): {calculate_mean_max_min(dataset_sub, 'pct_counts_mt')}")

    dataset_sub.obs["filter_pass"] = (
        (dataset_sub.obs["n_genes_by_counts"] >= 200)
        #(dataset_sub.obs["n_genes_by_counts"] <= 10000) &
        #(dataset_sub.obs["total_counts"] <= 5e5) &
        #(dataset_sub.obs["pct_counts_mt"] <= 30)
    ).astype(bool)

    #le = LabelEncoder()
    dataset_sub.obs['batch'] = dataset_sub.obs['_scvi_batch']
    dataset_sub.obs['str_labels'] = dataset_sub.obs['cell_ontology_class']
    dataset_sub.obs['celltype'] = dataset_sub.obs['cell_ontology_class']
    #dataset_sub.obs['labels'] = le.fit_transform(dataset_sub.obs['str_labels'])
    #label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    #print("Mapping:", label_mapping)
    dataset_sub.obs['n_counts'] = np.asarray(dataset_sub.X.sum(axis=1)).ravel() #dataset_sub.X.sum(axis=1)

    #print(f"Gene symbol: {dataset_sub.var['gene_symbol'][:50]}")
    #print(f"Ensemble id: {dataset_sub.var['ensembl_id'][:50]}")

    f_ensembl_ids = dataset_sub.var['ensembl_id'].str.replace(r'\.\d+$','', regex=True)

    #print(f"Formatted Ensemble id: {f_ensembl_ids[:50]}")
    
    dataset_sub.var['feature_id'] = f_ensembl_ids #dataset_sub.var['gene_symbol']
    dataset_sub.var['ensembl_id'] = f_ensembl_ids #dataset_sub.var['gene_symbol']

    print(f"sp.issparse(dataset_sub.X): {str(sp.issparse(dataset_sub.X))}")
    if sp.issparse(dataset_sub.X):
        n_counts = np.asarray(dataset_sub.X.sum(axis=1)).ravel()
        n_genes  = dataset_sub.X.getnnz(axis=1)
    else:
        n_counts = dataset_sub.X.sum(axis=1)
        n_genes  = (dataset_sub.X > 0).sum(axis=1)

    dataset_sub.obs["n_genes"]  = np.asarray(n_genes, dtype=np.int32)

    tk = LangCellTranscriptomeTokenizer(dict([(k, k) for k in dataset_sub.obs.keys()]), nproc=4)
    tokenized_cells, cell_metadata = tk.tokenize_anndata(dataset_sub)
    tokenized_dataset = tk.create_dataset(tokenized_cells, cell_metadata)

    print(tokenized_dataset)

    print(tokenized_dataset.column_names)
    print("------------------")
    print("All labels")
    print(list(set(dataset_sub.obs['str_labels'])))
    print("------------------")
    print(list(set(tokenized_dataset["celltype"])))

    tokenized_dataset.save_to_disk(f'../data/tabula_sapiens/{name_without_ext}.tokenized_dataset')

    #print("-----------------Newly created tokenized dataset ----------------------")
    #print(tokenized_dataset.column_names)
    #print(tokenized_dataset["labels"][:5], list(set(tokenized_dataset["labels"])))
    #print(len(tokenized_dataset["input_ids"]), len(tokenized_dataset["input_ids"][0]), len(tokenized_dataset["input_ids"][1]), tokenized_dataset["str_labels"][:5], tokenized_dataset["length"][:5], tokenized_dataset["filter_pass"][:5])
    #print(tokenized_dataset["n_counts"][:5], tokenized_dataset["n_genes"][:5], tokenized_dataset["str_labels"][:5], list(set(tokenized_dataset["str_labels"])))

    for label_name in ["celltype", "cell_type", "str_labels", "labels"]:
        if label_name in tokenized_dataset.column_names:
            break
    if label_name != "celltype":
        tokenized_dataset = tokenized_dataset.rename_column(label_name,"celltype")
    tokenized_dataset = tokenized_dataset.filter(lambda example: example['celltype'] != 'Other')

    #name_without_ext = "pbmc10k"
    type2text = json.load(open(f'../data/tablula_sapiens_datasets/type2text_{name_without_ext}.json'))
    #type2text = json.load(open('../data/data_zeroshot/type2text.json'))
    types = list(set(tokenized_dataset['celltype']))
    texts = texts = [type2text[typename] for typename in types] #[gettextfromname(typename) for typename in types]
    type2num = dict([(type, i) for i, type in enumerate(types)])

    print("-------------------------------------------------")
    print("Types:", types)
    print("-------------------------------------------------")
    print("Texts:", texts)
    print("-------------------------------------------------")
    print("type2num:", type2num)
    print("-------------------------------------------------")
    #
    def classes_to_ids(example):
        example["label"] = type2num[example["celltype"]]
        return example
    dataset = tokenized_dataset.map(classes_to_ids, num_proc=16)
    dataset = dataset.remove_columns(['length']) #'celltype', 

    dataset = dataset.shuffle(seed)

    print(dataset)

    # split
    label_num = len(type2num.keys())
    type2trainlist = {}
    for i in range(label_num):
        type2trainlist[i] = []
    if nshot >= 1:
        for i, l in enumerate(dataset["label"]):
            if len(type2trainlist[l]) < nshot:
                type2trainlist[l].append(i)
                br = True
                for k in type2trainlist.keys():
                    if len(type2trainlist[k]) < nshot:
                        br = False
                        break
                if br:
                    break
    print("type2trainlist:", type2trainlist)
    train_idx = []
    for k in type2trainlist.keys():
        train_idx += type2trainlist[k]
    test_idx = list(set(range(len(dataset))) - set(train_idx))
    # to keep: ['input_ids', 'n_counts', 'batch', 'labels', 'n_genes', 'filter_pass', 'label']
    features_to_keep = ['input_ids', 'n_counts', 'batch', 'labels', 'n_genes', 'filter_pass', 'label', 'celltype']
    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in features_to_keep])

    print("Only features to keep")
    print(dataset)

    traindataset = dataset.select(train_idx).shuffle(seed)
    testdataset = dataset.select(test_idx).shuffle(seed)
    testdataset_w_ct = testdataset

    traindataset = traindataset.remove_columns(['celltype'])
    testdataset = testdataset.remove_columns(['celltype'])

    print("Train dataset")
    print(traindataset)
    print("Test dataset")
    print(testdataset)
    print("Test dataset with celltypes")
    print(testdataset_w_ct)

    # traindataset, train_ind = extract_data_based_on_class(dataset, train_cls)
    # testdataset, test_ind = extract_data_based_on_class(dataset, test_cls)

    train_loader = DataLoader(traindataset, batch_size=train_batchsize, 
                            collate_fn=DataCollatorForCellClassification(), shuffle=False)
    test_loader = DataLoader(testdataset, batch_size=test_batchsize,
                            collate_fn=DataCollatorForCellClassification(), shuffle=False)
    eval_loader = DataLoader(testdataset.select(range(eval_num)), batch_size=test_batchsize,
                            collate_fn=DataCollatorForCellClassification(), shuffle=False)

    return train_loader, test_loader, texts, testdataset, testdataset_w_ct, tokenizer, text_encoder, types, name_without_ext

def fine_tune_model(model, train_loader, test_loader, texts, testdataset, testdataset_w_ct, text_encoder, types, name_without_ext):
    mode = "few_shot"
    model.train()
    text_encoder.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer1 = torch.optim.Adam(model.parameters(), lr=1e-5)
    optimizer2 = torch.optim.Adam(text_encoder.parameters(), lr=1e-5)

    for epoch in range(epochs):
        print('epoch:', epoch)
        for i, d in tqdm(enumerate(train_loader)):
            model.train()
            text_encoder.train()
            text_embs = torch.cat([text_encode(text) for text in texts], 0).T.cuda()
            cell_last_h, cellemb = cell_encode(d['input_ids'], d['attention_mask']) # batchsize * 256
            # text_embs: 256 * class_num
            sim = (cellemb @ text_embs) / 0.05 # batchsize * class_num
            loss_sim = loss_fn(sim, d['labels'].cuda())

            ctm_logit = torch.zeros_like(sim)
            for text_idx, text in enumerate(texts):
                text_list = [text] * sim.shape[0]
                ctm_logit[:, text_idx] = ctm(text_list, cell_last_h, d['attention_mask'])
            loss_ctm = loss_fn(ctm_logit, d['labels'].cuda())

            loss = loss_sim + loss_ctm
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()
    print("Fine tuning finished")
    ## Make inferences
    cell_embs = torch.zeros(len(testdataset), 256)
    model.eval()
    text_encoder.eval()
    preds = torch.zeros(len(testdataset))
    sim_logits = torch.zeros(len(testdataset), text_embs.shape[-1])
    ctm_logits = torch.zeros(len(testdataset), text_embs.shape[-1])
    logits = torch.zeros(len(testdataset), text_embs.shape[-1])
    labels = torch.tensor(testdataset['label'])
    text_embs = torch.cat([text_encode(text) for text in texts], 0).T.cuda()
    with torch.no_grad():
        for i, d in tqdm(enumerate(test_loader)):
            cell_last_h, cellemb = cell_encode(d['input_ids'], d['attention_mask']) # batchsize * 256
            sim = (cellemb @ text_embs) / 0.05 # batchsize * class_num
            sim_logit = F.softmax(sim, dim=-1)

            # ctm
            ctm_logit = torch.zeros_like(sim_logit)
            for text_idx, text in enumerate(texts):
                text_list = [text] * sim_logit.shape[0]
                ctm_logit[:, text_idx] = ctm(text_list, cell_last_h, d['attention_mask'])
            ctm_logit = F.softmax(ctm_logit, dim=-1)

            sim_logits[i * test_batchsize: (i + 1) * test_batchsize] = sim_logit.cpu()
            ctm_logits[i * test_batchsize: (i + 1) * test_batchsize] = ctm_logit.cpu()
            logit = (sim_logit + ctm_logit) / 2
            pred = logit.argmax(dim=-1)
            logits[i * test_batchsize: (i + 1) * test_batchsize] = logit.cpu()
            cell_embs[i * test_batchsize: (i + 1) * test_batchsize] = cellemb.cpu()
            preds[i * test_batchsize: (i + 1) * test_batchsize] = pred.cpu()

    torch.save({'cell_embs': cell_embs, 'text_embs': text_embs, 
                'sim_logits': sim_logits, 'ctm_logits': ctm_logits,
                'preds': preds, 'labels': labels, 'logits': logits}, 
            output_path + 'result.pt')

    from sklearn.metrics import f1_score, accuracy_score

    for k in [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        preds_k = (k * sim_logits + (1 - k) * ctm_logits).argmax(dim=-1)
        print(k, '\n', accuracy_score(labels, preds_k), f1_score(labels, preds_k, average='macro'))

    sim_preds = sim_logits.argmax(dim=-1)
    ctm_preds = ctm_logits.argmax(dim=-1)
    alpha = 0.1
    preds = (alpha * sim_logits + (1 - alpha) * ctm_logits).argmax(dim=-1)
    labels = torch.tensor(testdataset['label'])
    for row in confusion_matrix(labels, preds):
        print('\t'.join([str(x) for x in row]))
    print(classification_report(labels, preds, digits=4))
    organ_name = name_without_ext.split('_')[0]
    plot_confusion_matrix(labels, preds, types, title=f"Confusion matrix for celltypes of {organ_name}", normalize=True, \
                          file_n=name_without_ext, mode=mode)

    plt.tight_layout()
    cell_embs_ad = ad.AnnData(cell_embs.numpy())
    cell_embs_ad.obs['celltype'] = testdataset_w_ct['celltype']
    if 'batch' in testdataset.features.keys():
        cell_embs_ad.obs['batch'] = testdataset['batch']
        cell_embs_ad.obs['batch'] = cell_embs_ad.obs['batch'].astype(str)
    cell_embs_ad.obs['predictions'] = [types[i] for i in preds]
    sc.pp.neighbors(cell_embs_ad, use_rep='X', n_neighbors=80)
    sc.tl.umap(cell_embs_ad)
    sc.pl.umap(cell_embs_ad, color=['celltype', 'predictions', 'batch'], legend_fontsize ='xx-small', size=5, legend_fontweight='light')
    plt.savefig(f"../data/umap_plot_{name_without_ext}_{mode}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues, file_n=None, mode=None):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=False, cmap=cmap)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.show()
    plt.tight_layout()
    plt.savefig(f'../data/confusion_matrix_{file_n}_{mode}.png', dpi=300)

if __name__ == "__main__":

    #parser = argparse.ArgumentParser()
    #parser.add_argument("--data_path", type=str, default="/path/to/")
    #parser.add_argument("--output_path", type=str, default=None)
    #parser.add_argument("--epochs", type=int, default=10)
    #parser.add_argument("--train_batchsize", type=int, default=12)
    #parser.add_argument("--test_batchsize", type=int, default=64)
    #parser.add_argument("--nshot", type=int, default=1)
    #parser.add_argument("--seed", type=int, default=2024)
    #parser.add_argument("--device", type=int, default=0)
    #args = parser.parse_args()
    # model_path =  args.model_path
    #data_path = args.data_path
    #epochs = args.epochs
    #train_batchsize = args.train_batchsize
    #test_batchsize = args.test_batchsize
    #seed = args.seed
    #nshot = args.nshot
    output_path = '../data/output/ctm_' + str(nshot) + '-shot/'
    subprocess.call(f'mkdir {output_path}', shell=True)

    model, tokenizer, text_encoder, ctm_head = load_model()
    print("Loading model finished.")
    train_loader, test_loader, texts, testdataset, testdataset_w_ct, tokenizer, text_encoder, types, name_without_ext = load_dataset(tokenizer, text_encoder)
    print("Loading dataset finished.")
    fine_tune_model(model, train_loader, test_loader, texts, testdataset, testdataset_w_ct, text_encoder, types, name_without_ext)
