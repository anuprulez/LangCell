
import os
import scanpy as sc
import anndata as ad
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import scipy.sparse as sp
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from datasets import load_from_disk, Dataset
import torch.nn as nn, torch.nn.functional as F
import torch, json
from transformers import BertTokenizer, BertModel
from utils import BertModel as MedBertModel
from utils import LangCellDataCollatorForCellClassification as DataCollatorForCellClassification
from utils import LangCellTranscriptomeTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader

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

    print(ctm_head)
    return tokenizer, model, text_encoder, ctm_head

def text_encode(text, tokenizer, text_encoder):
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
    # TODO: Load AnnData
    #dataset = load_from_disk("../data/data_zeroshot/pbmc10k.dataset")
    #tokenized_dataset = dataset.shuffle(seed=42) #.select(range(1000))
    #print(tokenized_dataset.column_names)
    #print(tokenized_dataset["labels"][:5], list(set(tokenized_dataset["labels"])))
    #print(len(tokenized_dataset["input_ids"]), len(tokenized_dataset["input_ids"][0]), len(tokenized_dataset["input_ids"][1]), tokenized_dataset["str_labels"][:5], tokenized_dataset["length"][:5], tokenized_dataset["filter_pass"][:5])
    #print(tokenized_dataset["n_counts"][:5], tokenized_dataset["n_genes"][:5], tokenized_dataset["str_labels"][:5], list(set(tokenized_dataset["str_labels"])))

    #print("--------- END of Default data -----------------------")
    #print()


    data_dir = "../data/tablula_sapiens/Kidney_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad"
    dataset_sub = sc.read(data_dir)
    print(dataset_sub)
    #print()
    

    #sc.pp.normalize_total(dataset_sub, target_sum=1e4)

    # Logarithmize data
    #sc.pp.log1p(dataset_sub)

    # Feature selection
    #sc.pp.highly_variable_genes(dataset_sub, n_top_genes=500, min_mean=0.0125, max_mean=3, min_disp=0.5)
    #dataset_sub = dataset_sub[:, dataset_sub.var.highly_variable]
    #print("After filtering and normalization and feature selection")
    #print(tokenized_dataset)

    #print(dataset_sub.var['ensembl_id'][:5])
    #print("dataset_sub var mt")
    #print(dataset_sub.var["mt"][:5])

    sc.pp.filter_cells(dataset_sub, min_genes=200)
    sc.pp.filter_genes(dataset_sub, min_cells=3)
    sc.pp.highly_variable_genes(dataset_sub, n_top_genes=3000)
    dataset_sub = dataset_sub[:, dataset_sub.var.highly_variable]
    sc.pp.calculate_qc_metrics(dataset_sub, qc_vars=["mt"], inplace=True)

    print(dataset_sub)

    print(f"n_genes_by_counts (mean, med, max, min): {calculate_mean_max_min(dataset_sub, 'n_genes_by_counts')}")
    print(f"total_counts (mean, med, max, min): {calculate_mean_max_min(dataset_sub, 'total_counts')}")
    print(f"pct_counts_mt (mean, med, max, min): {calculate_mean_max_min(dataset_sub, 'pct_counts_mt')}")

    dataset_sub.obs["filter_pass"] = (
        (dataset_sub.obs["n_genes_by_counts"] >= 200) &
        (dataset_sub.obs["n_genes_by_counts"] <= 6000) &
        (dataset_sub.obs["total_counts"] <= 5e5) &
        (dataset_sub.obs["pct_counts_mt"] <= 20)
    ).astype(bool)

    le = LabelEncoder()
    dataset_sub.obs['batch'] = dataset_sub.obs['_scvi_batch']
    dataset_sub.obs['str_labels'] = dataset_sub.obs['cell_ontology_class']
    dataset_sub.obs['celltype'] = dataset_sub.obs['cell_ontology_class']
    dataset_sub.obs['labels'] = le.fit_transform(dataset_sub.obs['str_labels'])
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print("Mapping:", label_mapping)
    dataset_sub.obs['n_counts'] = np.asarray(dataset_sub.X.sum(axis=1)).ravel() #dataset_sub.X.sum(axis=1)

    print(f"Gene symbol: {dataset_sub.var['gene_symbol'][:50]}")
    print(f"Ensemble id: {dataset_sub.var['ensembl_id'][:50]}")

    f_ensembl_ids = dataset_sub.var['ensembl_id'].str.replace(r'\.\d+$','', regex=True)

    print(f"Formatted Ensemble id: {f_ensembl_ids[:50]}")
    
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

    print(list(set(tokenized_dataset["celltype"])))

    tokenized_dataset.save_to_disk('../data/tabula_sapiens/Kidney_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.tokenized_dataset')

    print("-----------------Newly created tokenized dataset ----------------------")
    print(tokenized_dataset.column_names)
    print(tokenized_dataset["labels"][:5], list(set(tokenized_dataset["labels"])))
    print(len(tokenized_dataset["input_ids"]), len(tokenized_dataset["input_ids"][0]), len(tokenized_dataset["input_ids"][1]), tokenized_dataset["str_labels"][:5], tokenized_dataset["length"][:5], tokenized_dataset["filter_pass"][:5])
    print(tokenized_dataset["n_counts"][:5], tokenized_dataset["n_genes"][:5], tokenized_dataset["str_labels"][:5], list(set(tokenized_dataset["str_labels"])))


    for label_name in ["celltype", "cell_type", "str_labels", "labels"]:
        if label_name in tokenized_dataset.column_names:
            break
    print(f"label_name: {label_name}")
    if label_name != "celltype":
        tokenized_dataset = tokenized_dataset.rename_column(label_name,"celltype")

    types = list(set(tokenized_dataset['celltype']))

    print(f"celltypes: {types}")
    print(f"labels: {list(set(tokenized_dataset['labels']))}")

    #type2text = json.load(open('../data/data_zeroshot/type2text.json'))
    type2text = json.load(open('../data/tablula_sapiens/type2text_Kidney_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.json'))

    print(type2text)

    #import sys
    #sys.exit()

    texts = [type2text[typename] for typename in types]
    with torch.no_grad():
        text_embs = torch.cat([text_encode(text, tokenizer, text_encoder) for text in texts], 0).T.cuda() # 256 * N
    text_embs.requires_grad = False
    type2num = dict([(type, i) for i, type in enumerate(types)])

    def classes_to_ids(example):
        example["label"] = type2num[example["celltype"]]
        return example

    testdataset = tokenized_dataset.map(classes_to_ids, num_proc=16)
    remove_columns = testdataset.column_names
    remove_columns.remove('input_ids')
    remove_columns.remove('label')
    testdataset = testdataset.remove_columns(remove_columns)
    batchsize = 32
    collator = DataCollatorForCellClassification()
    dataloader = DataLoader(testdataset, batch_size=batchsize, collate_fn=collator, shuffle=False)
    return tokenized_dataset, testdataset, texts, text_embs, dataloader, batchsize, types


def predict(dataset_sub, testdataset, texts, text_embs, dataloader, batchsize, types):
    cell_embs = torch.zeros(len(dataset_sub), 256)
    model.eval()
    text_encoder.eval()
    preds = torch.zeros(len(dataset_sub))
    sim_logits = torch.zeros(len(dataset_sub), text_embs.shape[-1])
    ctm_logits = torch.zeros(len(dataset_sub), text_embs.shape[-1])
    logits = torch.zeros(len(dataset_sub), text_embs.shape[-1])
    labels = torch.tensor(testdataset['label'])
    with torch.no_grad():
        for i, d in tqdm(enumerate(dataloader)):
            cell_last_h, cellemb = cell_encode(d['input_ids'], d['attention_mask']) # batchsize * 256
            sim = (cellemb @ text_embs) / 0.05 # batchsize * 161
            sim_logit = F.softmax(sim, dim=-1)

            # ctm
            ctm_logit = torch.zeros_like(sim_logit)
            for text_idx, text in enumerate(texts):
                text_list = [text] * sim_logit.shape[0]
                ctm_logit[:, text_idx] = ctm(text_list, cell_last_h, d['attention_mask'])
            ctm_logit = F.softmax(ctm_logit, dim=-1)

            logit = (sim_logit + ctm_logit) / 2
            pred = logit.argmax(dim=-1)
            sim_logits[i * batchsize: (i + 1) * batchsize] = sim_logit.cpu()
            ctm_logits[i * batchsize: (i + 1) * batchsize] = ctm_logit.cpu()
            logits[i * batchsize: (i + 1) * batchsize] = logit.cpu()
            cell_embs[i * batchsize: (i + 1) * batchsize] = cellemb.cpu()
            preds[i * batchsize: (i + 1) * batchsize] = pred.cpu()

        # torch.save({'cell_embs': cell_embs,
        #             'sim_logits': sim_logits, 'ctm_logits': ctm_logits, 
        #             'preds': preds, 'labels': labels, 'logits': logits}, 
        #            'data/results.pt')

    sim_preds = sim_logits.argmax(dim=-1)
    ctm_preds = ctm_logits.argmax(dim=-1)
    alpha = 0.1
    preds = (alpha * sim_logits + (1 - alpha) * ctm_logits).argmax(dim=-1)
    labels = torch.tensor(testdataset['label'])
    for row in confusion_matrix(labels, preds):
        print('\t'.join([str(x) for x in row]))
    print(classification_report(labels, preds, digits=4))
    plot_confusion_matrix(labels, preds, types, normalize=True)

    # Plot UMAP
    plt.tight_layout()
    cell_embs_ad = ad.AnnData(cell_embs.numpy())
    cell_embs_ad.obs['celltype'] = dataset_sub['celltype']
    if 'batch' in dataset_sub.features.keys():
        cell_embs_ad.obs['batch'] = dataset_sub['batch']
        cell_embs_ad.obs['batch'] = cell_embs_ad.obs['batch'].astype(str)
    cell_embs_ad.obs['predictions'] = [types[i] for i in preds]
    sc.pp.neighbors(cell_embs_ad, use_rep='X', n_neighbors=80)
    sc.tl.umap(cell_embs_ad)
    sc.pl.umap(cell_embs_ad, color=['celltype', 'predictions', 'batch'], legend_fontsize ='xx-small', size=5, legend_fontweight='light')
    plt.savefig("../data/umap_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
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
    plt.savefig('../data/confusion_matrix.png', dpi=300)


if __name__ == "__main__":
    tokenizer, model, text_encoder, ctm_head = load_model()
    print("Loading model finished.")
    dataset_sub, testdataset, texts, text_embs, dataloader, batchsize, types = load_dataset(tokenizer, text_encoder)
    print("Loading dataset finished.")
    predict(dataset_sub, testdataset, texts, text_embs, dataloader, batchsize, types)