import subprocess
from random import shuffle
import pandas as pd

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from .utils import *

def from_fasta_to_df(folder, file, chunksize=5000):
    ids, seqs, aligned_seqs = [], [], []
    with open(file, "r") as input_handle:
        for i, seq in enumerate(SeqIO.parse(input_handle, "fasta")):
            seq = seq.upper()
            aligned_seq = str(seq.seq)
            if "X" in aligned_seq:
                continue
            ids.append(seq.id)
            seq = "".join([c for c in aligned_seq if c in AA])
            seqs.append(seq), aligned_seqs.append(aligned_seq)
            print(f"Processing {i} sequences ...", end="\r")
    df = pd.DataFrame(index=ids)
    df["aligned_seq"] = aligned_seqs
    df["seq"] = seqs
    df["name"] = ids
    df["length"] = df.seq.apply(lambda seq: len(seq))
    return df

def from_df_to_fasta(folder, df, prefix=""):
    records_unaligned = []
    for i, data in enumerate(df.itertuples()):
        records_unaligned.append(SeqRecord(Seq(data.seq), id=str(data.name)))
    with open(f"{folder}/{prefix}unaligned.fasta", "w") as handle:
        SeqIO.write(records_unaligned, handle, "fasta")

def from_df_to_data(folder, df, prefix=""):
    N = len(df.aligned_seq.values[0])
    x = torch.zeros(len(df), 20, N)
    for i, x_ in enumerate(df.aligned_seq.values):
        x_ = torch.tensor([AA_IDS.get(aa, 20) for aa in x_])
        x_ = torch.tensor(to_onehot(x_, (None, 21)))
        x[i] = x_.t()[:-1]
    data = {"x": x, "L": len(df)}
    return data

def cluster_weights(folder):
    clusters = pd.read_table(f"{folder}/tmp/clusters.tsv_cluster.tsv", names=["cluster", "id"]).set_index("id").cluster
    cluster_weights = 1 / clusters.value_counts()
    weights = [cluster_weights[c] for c in clusters]
    return list(clusters.index), torch.tensor(list(weights))

def split_train_val_set(folder, ratio=0.1):
    clusters = pd.read_table(f"{folder}/tmp/clusters.tsv_cluster.tsv", names=["clusters", "id"]).set_index(
        "id").clusters
    max_size = ratio * len(clusters)
    val = []
    unique_clusters = list(clusters.unique())
    shuffle(unique_clusters)
    for c in unique_clusters:
        val += list(clusters[clusters == c].index)
        if len(val) > max_size:
            break
    is_val = torch.tensor([int(c in val) for c in clusters.index])
    subset = dict()
    subset["val"] = torch.where(is_val == 1)[0]
    subset["train"] = torch.where(is_val == 0)[0]
    return subset

def standardize_fasta(folder, file):
    df = from_fasta_to_df(folder, file)
    from_df_to_fasta(folder, df)
    data = from_df_to_data(folder, df)
    subprocess.run(
        f'mmseqs easy-cluster "{folder}/unaligned.fasta" "{folder}/tmp/clusters.tsv" "{folder}/tmp" --min-seq-id 0.3',
        shell=True)
    subprocess.run(
        f'mkdir "{folder}/tensorboard"',
        shell=True)
    subprocess.run(
        f'mkdir "{folder}/weights"',
        shell=True)
    data["cluster_index"], data["weights"] = cluster_weights(folder)
    data["subset"] = split_train_val_set(folder)
    torch.save(data, f"{folder}/data.pt")
