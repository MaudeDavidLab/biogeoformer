


import torch
from transformers import (
    AutoModel,
    EsmModel,
    EsmConfig,
    AutoModelForMaskedLM,
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    PreTrainedModel,
    PretrainedConfig,
    EsmForSequenceClassification,
    EsmTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    set_seed,
)
from evaluate import load
from torch.utils.data import Dataset, Subset, DataLoader
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
import datasets
from tqdm import tqdm
from prettytable import PrettyTable
import json
import os
import seaborn as sns
#import rmm
#from rmm.allocators.torch import rmm_torch_allocator
from accelerate import Accelerator
from accelerate.utils import convert_model
from torch.optim import Adafactor, AdamW
from transformers.trainer_utils import get_last_checkpoint
import gc
import time
import argparse
from typing import Union
from peft import LoraConfig, get_peft_model
from exp_log import append_experiment, base_row, flatten_metrics
from backbones import (
    BACKBONE_DIM_CKPT,
    BACKBONE_LAYER_CKPT,
    default_lora_adaptors,
    get_tokenizer,
    is_glm2,
    load_backbone,
    pool_exclude_ids,
    sequence_prefix,
)
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import random
from safetensors.torch import load_file
from pytorch_metric_learning.losses import SupConLoss
from pytorch_metric_learning.samplers import MPerClassSampler







# some globals
accuracy = load("accuracy")
matthews_metric = load("matthews_correlation")
f1_metric = load("f1")
precision_metric = load("precision")
recall_metric = load('recall')
conf_mat_metric = load("confusion_matrix")


# Kept as aliases: the tables now live in backbones.py so every backbone family
# (ESM2, gLM2) is registered in one place.
ESM_DIM_CKPT = BACKBONE_DIM_CKPT
ESM_LAYER_CKPT = BACKBONE_LAYER_CKPT




def init_mem_pool_for_grace():
    # INITIALIZE GH200 MEMORY MANAGEMENT
    # 1. Configure RMM for GH200 Unified Memory
    # 'managed_memory=True' is the critical flag that allows the GPU to 
    # transparently access the Grace CPU's huge RAM (480GB+).

    initial_size = 54 * 1024**3

    rmm.reinitialize(
        pool_allocator=True,
        managed_memory=True,  # <--- MUST ENABLE FOR GH200 OVERSUBSCRIPTION
        initial_pool_size=initial_size, # Let it grow dynamically
    )

    # 2. Hot-swap the PyTorch allocator
    # From this point on, every .to('cuda') or torch.tensor(..., device='cuda')
    # uses RMM instead of the default PyTorch caching allocator.
    torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

    print(f"Current Allocator: {torch.cuda.memory.get_allocator_backend()}")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision = precision_metric.compute(predictions=predictions, references=labels, average='weighted')
    recall = recall_metric.compute(predictions=predictions, references=labels, average='weighted')
    acc = accuracy.compute(predictions=predictions, references=labels)
    mcc = matthews_metric.compute(references=labels, predictions=predictions)    
    f1 = f1_metric.compute(references=labels, predictions=predictions, average='weighted')
    return {"accuracy": acc, "MCC": mcc, "F1": f1, 'precision': precision, 'recall': recall}


def print_model_size(model):
    total_params = model.num_parameters()
    trainable_params = model.num_parameters(only_trainable=True)
    
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    print(f"Trainable Ratio: {trainable_params/total_params:.2%}")


"""
Little function for printing our parameters/layer :D
"""
def layer_breakdown(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue # Skip frozen layers
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
        
    print(table)
    print(f"Total Trainable Params: {total_params}")




# CONFIG CLASSES

class ESMConfig(PretrainedConfig):
    model_type = "esm2"
    def __init__(
        self, 
        nu_labels=7000, 
        esm_ckpt='facebook/esm2_t6_8M_UR50D', 
        do_lora=False,
        lora_adaptors=None,
        lora_dropout=0.0,
        lora_rank=8,
        freeze_backbone=False,
        **kwargs,
        ):

        super().__init__(**kwargs)
        self.nu_labels         = nu_labels
        self.esm_ckpt          = esm_ckpt
        self.d_model           = ESM_DIM_CKPT[esm_ckpt]
        self.freeze_backbone   = freeze_backbone
        self.do_lora           = do_lora
        # ESM2 has separate q/k/v projections, gLM2 fuses them into 'wqkv'.
        self.lora_adaptors     = lora_adaptors or default_lora_adaptors(esm_ckpt)
        self.lora_dropout      = lora_dropout
        self.lora_rank         = lora_rank
        self.layer_idx         = ESM_LAYER_CKPT[esm_ckpt] // 2


################################






# MODEL CLASSES



"""
Generic classification head
"""
class MLP(torch.nn.Module):
    def __init__(self, output_size, input_size):
        super(MLP, self).__init__()
        self.l1 = torch.nn.Linear(input_size, output_size)
        self.norm = torch.nn.LayerNorm(input_size)
        #self.dropout = torch.nn.Dropout(0.0)
    def forward(self, x):
        h = self.norm(x)
        #h = self.dropout(h)
        h = self.l1(h)
        return h



class ESM2LoRAForSequenceClassification(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        config_class = ESMConfig

        
        backbone = load_backbone(config.esm_ckpt, config.layer_idx)


        lora_config = LoraConfig(
                r=config.lora_rank,                
                lora_alpha=config.lora_rank * 2,      
                target_modules=config.lora_adaptors, 
                lora_dropout=config.lora_dropout,
                bias="none",
        )

        self.backbone = get_peft_model(backbone, lora_config)

        #self.clf_head_norm = torch.nn.LayerNorm(config.d_model)
        self.clf_head = MLP(input_size=config.d_model, output_size=config.nu_labels)
        #self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean', label_smoothing=config.label_smoothing)
        self.freeze_backbone = config.freeze_backbone
        self.layer_idx = config.layer_idx
        tok = get_tokenizer(config.esm_ckpt)
        # Everything that is not a real residue (CLS/EOS for ESM2, the strand
        # and separator tokens for gLM2) is kept out of the mean pool.
        self.pool_exclude_ids = pool_exclude_ids(tok, config.esm_ckpt)

    def forward(self, input_ids=None, labels=None, attention_mask=None, token_type_ids=None):
        assert attention_mask is not None, "mean pooling requires an attention mask"

        h = self.backbone(
            input_ids,
            attention_mask=attention_mask,        # was missing
            output_hidden_states=True,
        ).last_hidden_state #s[self.layer_idx]           # [B, L, D]

        # True only on real residues. Pads are already excluded by attention_mask.
        pool_mask = attention_mask.bool()           # [B, L]
        for token_id in self.pool_exclude_ids:
            pool_mask &= (input_ids != token_id)

        pooled = self.mean_mask_pooling(h, pool_mask)
        return pooled, self.clf_head(pooled)

    """
    Simple function derived using Gemini. Use the attention mask in order to properly
    sum over the actual tokens in the batch and average them to do the mean pooling.
    """

    def mean_mask_pooling(self, h, pool_mask):
        """h: [B, L, D] bf16 | pool_mask: [B, L] bool -> [B, D]"""
        m = pool_mask.unsqueeze(-1).to(h.dtype)               # [B, L, 1]
        summed = (h * m).sum(dim=1).float()                   # [B, D], fp32
        counts = pool_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1], exact int64
        pooled = summed / counts
        return pooled.to(self.clf_head.norm.weight.dtype)



"""
Same backbone/pooling/head structure as ESM2LoRAForSequenceClassification, but with
no adapters: the backbone is either fully trained or fully frozen depending on
config.freeze_backbone. Use this for full fine-tuning or linear-probe style runs.
"""
class ESM2ForSequenceClassification(PreTrainedModel):
    config_class = ESMConfig

    def __init__(self, config):
        super().__init__(config)

        # load_backbone also removes the part of the model we do not need,
        # depending on where we are pulling the embedding from
        self.backbone = load_backbone(config.esm_ckpt, config.layer_idx)

        self.freeze_backbone = config.freeze_backbone
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        self.clf_head = MLP(input_size=config.d_model, output_size=config.nu_labels)
        self.layer_idx = config.layer_idx
        tok = get_tokenizer(config.esm_ckpt)
        # Everything that is not a real residue (CLS/EOS for ESM2, the strand
        # and separator tokens for gLM2) is kept out of the mean pool.
        self.pool_exclude_ids = pool_exclude_ids(tok, config.esm_ckpt)

    """
    Keep the frozen backbone in eval mode so dropout stays off even when the
    Trainer flips the top-level module into train mode.
    """
    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, input_ids=None, labels=None, attention_mask=None, token_type_ids=None):
        assert attention_mask is not None, "mean pooling requires an attention mask"

        with torch.set_grad_enabled(not self.freeze_backbone and torch.is_grad_enabled()):
            h = self.backbone(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            ).last_hidden_state                    # [B, L, D]

        # True only on real residues. Pads are already excluded by attention_mask.
        pool_mask = attention_mask.bool()           # [B, L]
        for token_id in self.pool_exclude_ids:
            pool_mask &= (input_ids != token_id)

        pooled = self.mean_mask_pooling(h, pool_mask)
        return pooled, self.clf_head(pooled)

    """
    Simple function derived using Gemini. Use the attention mask in order to properly
    sum over the actual tokens in the batch and average them to do the mean pooling.
    """

    def mean_mask_pooling(self, h, pool_mask):
        """h: [B, L, D] bf16 | pool_mask: [B, L] bool -> [B, D]"""
        m = pool_mask.unsqueeze(-1).to(h.dtype)               # [B, L, 1]
        summed = (h * m).sum(dim=1).float()                   # [B, D], fp32
        counts = pool_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1], exact int64
        pooled = summed / counts
        return pooled.to(self.clf_head.norm.weight.dtype)


# TODO
# CL objective
class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, backbone_lr=None, head_lr=None, label_smoothing, **kwargs):
        super().__init__(*args, **kwargs)

        self.class_weights = class_weights
        self.backbone_lr = backbone_lr
        self.head_lr = head_lr
        self.loss_fct_ce = torch.nn.CrossEntropyLoss(
            weight=self.class_weights, #.to(logits.device),
            label_smoothing=label_smoothing,
        )
        #self.loss_fct_cl = SupConLoss()


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        mean_pooled_proteins, logits = outputs

        loss = self.loss_fct_ce(logits, labels) # + (0.5 * self.loss_fct_cl(mean_pooled_proteins, labels))

        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name == 'clf_head':
                head_params.append(param)
            else:
                backbone_params.append(param)
        
        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": self.backbone_lr})
        if head_params:
            param_groups.append({"params": head_params, "lr": self.head_lr})

        self.optimizer = torch.optim.AdamW(
            param_groups,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.weight_decay,
        )

        return self.optimizer



# DATASETS

"""
From Gemini
"""
def balance_dataframe(df, label_col, n_samples, seed):
    """
    Resamples a dataframe so every label has exactly n_samples.
    If a label has < n_samples, it samples with replacement (upsampling).
    If a label has > n_samples, it samples without replacement (downsampling).
    """
    def sampler(group):
        # If we have enough samples, don't replace (downsample)
        # If we don't have enough, replace=True (upsample)
        return group.sample(n=n_samples, replace=len(group) < n_samples, random_state=seed)

    return df.groupby(label_col, group_keys=False).apply(sampler)




class SequenceClassificationDataset(Dataset):
    """
    Sequence classification dataset.

    Returns a tokenized sequence along with its label. Supports reading from
    a CSV file or a FASTA file whose headers follow the format:
        ><seq_id>|<KO_label>

    Args:
        csv_file:       Path to a CSV or FASTA file.
        label_col:      Column (or tag) name containing class labels.
        tokenizer:      HuggingFace tokenizer for sequence encoding.
        data_seed:      Random seed for sampling operations.
        seq_per_label:  If > 0, balance to this many sequences per label.
        median_sample:  If True, balance to the median per-label count.
        load_labels:    If True, load an existing label map from disk.
        label_map:      Path to save/load the label-to-index JSON mapping.
    """

    def __init__(
        self,
        csv_file,
        label_col,
        tokenizer,
        data_seed=42,
        seq_per_label=-1,
        median_sample=False,
        load_labels=False,
        label_map=None,
    ):
        if csv_file.split(".")[-1] == "fasta":
            print(f"Reading {csv_file}...")
            k_numbers, seqs, seq_ids = [], [], []
            for record in SeqIO.parse(csv_file, "fasta"):
                try:
                    seq_id = record.description.split("|")[0]
                    k_number = record.description.split("|")[1]
                    k_numbers.append(k_number)
                    seq_ids.append(seq_id)
                    seqs.append("".join(list(record.seq)))
                except IndexError:
                    print(f"Skipping malformed header: {record.description}")
            df = pd.DataFrame({"id": seq_ids, "sequence": seqs, "KO": k_numbers})
        else:
            df = pd.read_csv(csv_file)

        if median_sample:
            counts = df[label_col].value_counts()
            n_samples = int(counts.median())
            print(f"Balancing dataset to median count: {n_samples} per {label_col}")
            df = balance_dataframe(df=df, label_col=label_col, n_samples=n_samples, seed=data_seed)
        elif seq_per_label > 0:
            print(f"Balancing data so we have {seq_per_label} sequences per {label_col}")
            df = balance_dataframe(df=df, label_col=label_col, n_samples=seq_per_label, seed=data_seed)
        else:
            print("No sampling given")

        self.ids = list(df.id)
        self.sequences = list(df.sequence)
        self.labels = list(df[label_col])
        self.nu_labels = len(df[label_col].unique())
        self.tokenizer = tokenizer

        print(f"# of sequences for training: {len(self.sequences)}")
        print(f"# of labels: {self.nu_labels}")

        assert label_map is not None, (
            "A label_map path must be provided for both loading and saving."
        )
        if load_labels:
            with open(label_map, "r") as f:
                self.label_idx = json.load(f)
            print(f"Loaded label map from {label_map}")
        else:
            self.label_idx = dict(
                zip(list(df[label_col].unique()), range(self.nu_labels))
            )
            with open(label_map, "w") as f:
                json.dump(self.label_idx, f, indent=4)
            print(f"Saved label map to {label_map}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sample = self.tokenizer(
            self.sequences[idx],
            return_tensors="pt",
        )
        c = self.labels[idx]
        sample["id"] = self.ids[idx]
        sample["labels"] = self.label_idx[c]
        sample["label_str"] = c
        return sample



class CycDataset(Dataset):
    def __init__(self, csv_file, tokenizer, data_seed=42, seq_per_label=-1, load_labels=False, label_map=None, eval_set=False, seq_prefix=""):
        df = pd.read_csv(csv_file)

        if seq_per_label > 0:
            print(f"Balancing data so we have {seq_per_label} sequences per class")
            df = balance_dataframe(df, label_col='cycle', n_samples=seq_per_label, seed=data_seed)
            #df = df.groupby('KO', group_keys=False).apply(lambda x: x.sample(seq_per_label))

        self.sequences = list(df.sequence)
        self.labels = list(df.cycle)

        unique_labels = np.unique(self.labels)

        # 1. Get the counts of each label from your list
        label_counts = Counter(self.labels)
        label_idx = {str(label): i for i, label in enumerate(unique_labels)}

        self.class_weights = compute_class_weights(
            label_counts=label_counts, 
            label_idx=label_idx, 
            device='cuda' if torch.cuda.is_available() else 'cpu',
            max_ratio=10.0
        )
        
        self.nu_labels = len(df.cycle.unique())
        self.tokenizer = tokenizer
        # gLM2 needs a strand token in front of every protein; empty for ESM2.
        self.seq_prefix = seq_prefix

        print(f"# of sequences for training: {len(self.sequences)}")
        print(f"# of labels: {self.nu_labels}")

        if load_labels:
            assert label_map != None, "Indicated label map LOADING but no label map path provided."
            with open(label_map, "r") as f:
                self.label_idx = json.load(f)
            print(f"Loaded label map to {label_map}")
        else:
            assert label_map != None, "Indicated label map SAVING but no label map path provided."
            self.label_idx = dict(zip(list(df.cycle.unique()), [i for i in range(self.nu_labels)]))
            with open(label_map, "w") as f:
                json.dump(self.label_idx, f, indent=4)
            print(f"Saved label map to {label_map}")
        

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):

        sample = self.tokenizer(
            self.seq_prefix + self.sequences[idx],
        )

        c = self.labels[idx]
        sample['labels'] = self.label_idx[c]
        return sample


##################################


# experiments
###########################
"""
Code for classification. 

"""
def train(
    esm_ckpt,
    id,
    freeze_backbone,
    tr_data,
    data_seed,
    seq_per_label,
    model_dir, 
    total_steps, 
    label_smoothing,
    head_lr,
    backbone_lr,
    warmup_ratio,
    beta_1,
    beta_2,
    weight_decay,
    epochs,
    eval_data=None,
    eval_only=False,
    max_grad_norm=2.0, 
    bs=16,
    grad_accum_steps=8,
    print_params=False,
    model_ckpt=None,
    do_lora=False,
    lora_rank=8,
    exp_log=None,
    notes=None,
    ):

    # Snapshot every argument this run was called with (must stay the first
    # statement so locals() holds nothing but the arguments).
    run_config = dict(locals())
    run_config.pop("exp_log", None)
    run_config["effective_bs"] = bs * grad_accum_steps
    run_started = time.time()


    # Set global seed
    set_seed(data_seed)
    print(f"Seeds set to: {data_seed}")


    # Load in data. When resuming from a checkpoint the backbone family is
    # whatever that run was trained with, not whatever --esm_ckpt defaults to.
    backbone_ckpt = ESMConfig.from_pretrained(model_ckpt).esm_ckpt if model_ckpt else esm_ckpt
    print(f"Backbone: {backbone_ckpt} ({'gLM2' if is_glm2(backbone_ckpt) else 'ESM2'})")
    run_config["backbone_ckpt"] = backbone_ckpt

    tokenizer = get_tokenizer(backbone_ckpt)
    seq_prefix = sequence_prefix(backbone_ckpt)

    ds = CycDataset(
        tr_data,
        tokenizer=tokenizer, 
        data_seed=data_seed, 
        seq_per_label=seq_per_label,
        load_labels=True,
        label_map=f'../data/cyc_id_{id}_label_map.json',
        seq_prefix=seq_prefix,
    )

    class_weights = ds.class_weights

    ds_val = CycDataset(
        eval_data,
        tokenizer=tokenizer,
        data_seed=data_seed,
        load_labels=True,
        seq_per_label=seq_per_label,
        label_map=f'../data/cyc_id_{id}_label_map.json',
        eval_set=True,
        seq_prefix=seq_prefix,
    )

    # Calculate number of labels
    nu_labels = ds.nu_labels

    run_config.update(
        n_train=len(ds),
        n_val=len(ds_val),
        nu_labels=nu_labels,
    )

    # load in model   
    if model_ckpt:
        print(f"loading ckpt: {model_ckpt}")
        config = ESMConfig.from_pretrained(model_ckpt)
        model_cls = ESM2LoRAForSequenceClassification if do_lora else ESM2ForSequenceClassification
        model = model_cls(config)
        state = load_file(os.path.join(model_ckpt, "model.safetensors"))
        model.load_state_dict(state, strict=True)
    elif do_lora:
        print(f"Loading in {esm_ckpt} with {nu_labels} outputs for classification using LoRa")
        config = ESMConfig(
            nu_labels=nu_labels, 
            esm_ckpt=esm_ckpt, 
            freeze_backbone=freeze_backbone,
            #label_smoothing=label_smoothing,
            do_lora=do_lora,
            lora_rank=lora_rank,
        )
        print(config)
        model = ESM2LoRAForSequenceClassification(config)
    else:
        mode = "frozen backbone" if freeze_backbone else "full fine-tune"
        print(f"Loading in {esm_ckpt} with {nu_labels} outputs for classification ({mode})")
        config = ESMConfig(
            nu_labels=nu_labels,
            esm_ckpt=esm_ckpt,
            freeze_backbone=freeze_backbone,
            do_lora=do_lora,
        )
        print(config)
        model = ESM2ForSequenceClassification(config)



    if print_params: layer_breakdown(model)


    # run on trainer
    tr_args = TrainingArguments(
        output_dir=model_dir,
        #data_seed=data_seed,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        adam_beta1=beta_1,
        adam_beta2=beta_2,
        num_train_epochs=1,
        #max_steps=total_steps,
        eval_strategy='no',
        lr_scheduler_type='cosine',
        logging_strategy='steps',
        logging_steps=100,
        eval_steps=total_steps,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        dataloader_num_workers=2,
        local_rank=-1,
        include_num_input_tokens_seen=True,
        disable_tqdm=False,
        save_strategy='epoch',
        max_grad_norm=max_grad_norm,
        bf16=False,
        gradient_accumulation_steps=grad_accum_steps,
        #train_sampling_strategy='group_by_length',
        #torch_compile=True,
    )

    # should manage padding/batching automatically
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding='longest', # <-- pad to longest in the batch 
        return_tensors='pt',
        pad_to_multiple_of=8,
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=tr_args,
        class_weights=class_weights,
        data_collator=data_collator,
        train_dataset=ds,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics,
        backbone_lr=backbone_lr,
        head_lr=head_lr,
        label_smoothing=label_smoothing,
    )

    run_config.update(
        total_params=model.num_parameters(),
        trainable_params=model.num_parameters(only_trainable=True),
    )

    try:
        if eval_only: # and ko_eval_data != None:
            print("RUNNING INFERENCE ONLY")
            results = trainer.evaluate()
            print(results)
            log_run(exp_log, run_config, run_started, "eval_only",
                    eval_metrics=results, notes=notes)
        else:
            train_result = trainer.train()
            print(f"Model trained with batch size of {trainer.args.per_device_train_batch_size}")
            results = trainer.evaluate()
            print(results)
            log_run(exp_log, run_config, run_started, "ok",
                    train_metrics=train_result.metrics, eval_metrics=results,
                    notes=notes)
    except Exception as e:
        log_run(exp_log, run_config, run_started, "failed",
                notes=notes, error=f"{type(e).__name__}: {e}")
        raise


"""
Append this run to the experiment CSV. Logging must never take down a run that
otherwise finished, so any failure here is reported and swallowed.
"""
def log_run(exp_log, run_config, run_started, status, train_metrics=None,
            eval_metrics=None, notes=None, error=None):
    if not exp_log:
        return
    try:
        row = base_row(status=status, started_at=run_started, notes=notes)
        row.update(run_config)
        if train_metrics:
            row["train_loss"] = train_metrics.get("train_loss")
        if eval_metrics:
            skip = {"eval_runtime", "eval_samples_per_second",
                    "eval_steps_per_second", "epoch"}
            row.update({k: v for k, v in flatten_metrics(eval_metrics).items()
                        if k not in skip})
        if error:
            row["error"] = error
        append_experiment(exp_log, row)
    except Exception as e:
        print(f"[exp_log] WARNING: failed to record run: {type(e).__name__}: {e}")



def scratch():
    model = AutoModelForMaskedLM.from_pretrained("../LC-PLM", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    sequence = "MKTLLLTLLVVTIVCLDLGYSLKCYQHGKVVTCHRDMKFCYHNTGMPFRNLKLILQGCSSSCSETENNKCCSTDRCNK"
    inputs = tokenizer(sequence, return_tensors="pt")
    device = torch.device("cuda:0")
    model = model.to(device)
    print_model_size(model)


    with torch.no_grad():
        outputs = model(**inputs.to(device), output_hidden_states=True)

    last_hidden_state = outputs.hidden_states[-1]
    print(last_hidden_state.shape)

    construct_combined_tokenizer()
    





def plot_eval_metrics(trainer_state: Union[dict, str], save_path: str = "eval_metrics.png"):
    """Plot all eval metrics from a HuggingFace trainer_state.json"""
    
    if isinstance(trainer_state, str):
        with open(trainer_state) as f:
            trainer_state = json.load(f)
    
    # Extract eval entries only
    eval_logs = [log for log in trainer_state["log_history"] if "eval_loss" in log]
    
    if not eval_logs:
        print("No eval metrics found in trainer state.")
        return
    
    exclude = {"epoch", "step", "eval_runtime", "eval_samples_per_second", "eval_steps_per_second"}

    def flatten_log(log: dict, prefix: str = "") -> dict:
        """Recursively flatten nested dicts into dot-separated keys."""
        flat = {}
        for k, v in log.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(flatten_log(v, prefix=full_key))
            elif isinstance(v, (int, float)):
                flat[full_key] = v
        return flat

    # Flatten all logs
    flat_logs = [flatten_log(log) for log in eval_logs]

    # Collect all metric keys across all logs (excluding runtime stats)
    all_keys = set()
    for log in flat_logs:
        all_keys.update(log.keys())
    metric_keys = sorted(k for k in all_keys if k not in exclude)

    steps = [log["step"] for log in eval_logs]

    fig, axes = plt.subplots(len(metric_keys), 1, figsize=(8, 3 * len(metric_keys)))
    if len(metric_keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, metric_keys):
        # Use None for missing values at certain steps
        values = [flat_log.get(key, None) for flat_log in flat_logs]
        ax.plot(steps, values, marker="o")
        ax.set_title(key)
        ax.set_xlabel("Step")
        ax.set_ylabel(key)
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved to {save_path}")




def compute_class_weights(label_counts, label_idx, device='cpu', max_ratio=10.0):
    """
    Sqrt-dampened inverse-frequency class weights.

    Raw inverse frequency can produce extreme ratios (e.g. 120x) for
    imbalanced datasets, destabilizing training. Sqrt dampening compresses
    the range so rare classes get moderate upweighting (typically 2-5x)
    instead of extreme values. Unlike log, sqrt is always positive.

    Additionally normalizes so min weight = 1.0 and caps at max_ratio.
    """
    n_classes = len(label_idx)
    total = sum(label_counts.values())
    weights = torch.zeros(n_classes)
    for label_str, idx in label_idx.items():
        count = label_counts.get(label_str, label_counts.get(
            int(label_str) if label_str.isdigit() else label_str, 1))
        # Sqrt dampening: always positive, compresses extreme ratios
        weights[idx] = np.sqrt(total / (n_classes * count))
    # Normalize so min weight = 1.0, then cap
    weights = weights / weights.min()
    weights = torch.clamp(weights, max=max_ratio)
    print(f'Class weight range: {weights.min():.2f} to {weights.max():.2f} (ratio: {weights.max()/weights.min():.1f}x)')
    return weights.to(device)



def parse_args():
    parser = argparse.ArgumentParser(description="Train ESM2-based model")

    parser.add_argument("--esm_ckpt", type=str, default="facebook/esm2_t30_150M_UR50D",
                        help="Backbone checkpoint. ESM2 (facebook/esm2_*) or gLM2 "
                             "(tattabio/gLM2_150M, tattabio/gLM2_650M); see backbones.py.")
    parser.add_argument("--id", type=int, default=90)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--no_freeze_backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--tr_data", type=str, default="../data/train/bgf_train_60.csv")
    parser.add_argument("--eval_data", type=str, default="../data/val/bgf_val_60.csv")
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--seq_per_label", type=int, default=-1)
    parser.add_argument("--model_dir", type=str, default="../models/bgf_60_150_frozen")
    parser.add_argument("--total_steps", type=int, default=2500)
    #parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--beta_1", type=float, default=0.9)
    parser.add_argument("--beta_2", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--print_params", action="store_true", default=False)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--lora", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--head_lr", type=float, default=5e-4)
    parser.add_argument("--backbone_lr", type=float, default=5e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--exp_log", type=str, default="../experiments.csv",
                        help="CSV that every run appends one row to. Pass '' to disable.")
    parser.add_argument("--notes", type=str, default="",
                        help="Free-text note stored with this run in --exp_log.")

    return parser.parse_args()


if __name__ == "__main__":
    
    args = parse_args()
    train(
        esm_ckpt=args.esm_ckpt,
        id=args.id,
        eval_only=args.eval_only,
        freeze_backbone=args.freeze_backbone,
        label_smoothing=args.label_smoothing,
        head_lr=args.head_lr,
        backbone_lr=args.backbone_lr,
        tr_data=args.tr_data,
        eval_data=args.eval_data,
        data_seed=args.data_seed,
        seq_per_label=args.seq_per_label,
        model_dir=args.model_dir,
        total_steps=args.total_steps,
        warmup_ratio=args.warmup_ratio,
        beta_1=args.beta_1,
        beta_2=args.beta_2,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        max_grad_norm=args.max_grad_norm,
        bs=args.bs,
        grad_accum_steps=args.grad_accum_steps,
        print_params=args.print_params,
        model_ckpt=args.model,
        do_lora=args.lora,
        lora_rank=args.lora_rank,
        exp_log=args.exp_log,
        notes=args.notes,
    )
    




