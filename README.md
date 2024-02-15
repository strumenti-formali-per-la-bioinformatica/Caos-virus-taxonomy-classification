# Table of Contents

* [Introduction](#introduction)
* [Requirements](#requirements)
  * [Dataset](#dataset)
* [Usage](#usage)
* [Results](#results)
  * [How to reproduce the experiment](#how-to-reproduce-the-experiment)
* [Citations of works used](#citations-of-works-used)

# Introduction

The study of taxonomic classification of viruses is essential for the accurate identification of viruses, 
diagnosis of diseases, monitoring of viral evolution, and conservation of biodiversity. 
This knowledge is critical for protecting human, animal, and environmental health and for developing 
effective strategies for prevention, control, and treatment of viral diseases.

In the study of taxonomic classification of viruses, a key tool is alignment systems. These tools allow the genetic 
sequences of viruses to be compared in order to identify similarities and differences between them.

However, the alignment of viral sequences presents some problems. Viruses can differ significantly in 
their genomic structure, size and complexity. Some viruses may have sequences that are highly mutated, 
fragmented, or have variable regions. These characteristics make it difficult to achieve accurate 
sequence alignment. 

With the advent of modern sequencing technologies, genomic databases are expanding rapidly, 
with thousands of sequences being deposited regularly. Therefore, efficient and automated methods are 
required to manage and analyze these huge amounts of data.

To address these challenges, more and more researchers are adopting machine learning approaches.

Our project proposes an approach to the problem of taxonomic classification of viruses using machine learning. 
More specifically, our approach is based on graph classification using Graph Neural Networks.

The idea is to represent the reads to be classified in the form of graphs, using De Bruijn graphs 
and then use a GNN for their classification.

# Requirements

The requirements for using the project are as follows:
* Unix-like operating system.
* Anaconda environment with version of Python at least 3.8.

The following programs must be installed for the program to work properly:
- **ncbi-acc-download**: necessary to be able to download the complete genome of a given virus from the accession 
ids contained within the file ```data/sequence_identifiers.txt```.
- **genometools**: allows you to generate reads without or with overlap from complete genomes.
- **cd-hit**: allows clustering of sequences is then eliminates those reads that are too similar to each other.

Following are the commands to install these tools

```shell
sudo apt install ncbi-acc-download
sudo apt install genometools
sudo apt install cd-hit
```

The following are the commands to install torch with cuda support and the torch geometric library.

```shell
# Install torch and torch geometric with cuda support
CUDA_VERSION=cu118
TORCH_VERSION=2.0.0
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${CUDA_VERSION}
pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
pip3 install torch_geometric
```

The following are the commands to install taxonomy data from NCBI

```shell
USER=UsernameFLaTNNBio
mkdir /home/${USER}/.taxonkit
wget ftp://ftp.ncbi.nih.gov/pub/taxonomy/taxdump.tar.gz -P /home/${USER}/.taxonkit
tar -xvzf /home/${USER}/.taxonkit/taxdump.tar.gz -C /home/${USER}/.taxonkit
```

Finally, below are the commands to install all the remaining libraries.

```shell
conda install -c anaconda networkx
conda install -c conda-forge matplotlib
conda install -c conda-forge biopython
conda install -c conda-forge xmltodict
conda install -c bioconda pytaxonkit
conda install -c conda-forge einops
conda install -c conda-forge tabulate
conda install -c conda-forge tqdm
conda install -c anaconda scikit-learn
conda install -c pytorch pytorch
```

## Dataset

In order to download the data used during our experimentation, it is necessary to run the script 
```data/download_complete_genome.py```. The script does nothing more than download the genomes of all 
*accession ids* specified within the file ```data/sequence_identifiers.txt```.

An api key must be provided in order to use this script.
[Click here to obtain an api key](https://support.nlm.nih.gov/knowledgebase/article/KA-05317/en-us).

```shell
API_KEY=your_api_key
python3 data/download_complete_genome.py ${API_KEY}
```

Once the various necessary fasta files have been downloaded, through the ```pre_processing``` script the *training*, 
*validation* and *test* datasets can be generated.
In order to use such a script, it is necessary to enter the email associated with the NCBI account.

```shell
EMAIL=your_email
python3 pre_processing.py ${EMAIL}
```

# Usage

Once we have downloaded the data and run the pre-processing script, we are ready to be able to train and 
evaluate our model.

In order to train and evaluate the model, it is necessary to run the ```main.py``` script.
The following is a description of the different inputs.

* ```len_read```: define length of reads.
* ```len_overlap```: define length of overlapping between reads.
* ```k_size```: define length of kmers.
* ```batch_size```: define the batch size.
* ```model```: select the model to use. The possible values are:
  * *diff_pool*: select as a DiffPool model.
  * *ug_gat*: selects as a UGformer model that uses a GAT layer as the convolutional layer.
  * *ug_gcn*: selects as a UGformer model that uses a GCN layer as the convolutional layer.
* ```hidden_size```: defines the number of neurons to be used in the hidden layers.
* ```n_layers```: defines the number of layers.
* ```embedding```: defines the size of the embeddings. 
* ```embedding_mlp```: defines the size of the embeddings in the fully connected layers. 
* ```n_att_layers```: defines the number of attention layers.
* ```tf_heads```: defines the number of heads used by the transformers.
* ```gat_heads```: defines the number of heads used by the gat layers.

# Results

This section shows the results obtained.

The experiments were carried out by setting the read size to 250bp with overlap of 200bp.

Two different models, more specifically **UGformer** and **DiffPool**, were tested during the various 
experiments. References to these two models can be found in the "*[Citations of works used](#citations-of-works-used)*" 
section. 

In addition to testing the different hyperparameters, the size of the kmer, ```len_kmer``` parameter, was also tested, 
varying it from a size of 3 to a maximum of 19. The best result was obtained by setting the ```len_kmer``` parameter to 14.

The DiffPool model is the one that obtained better results. 
By grid search, the hyperparameters that yielded the best result are as follows:

```shell
+-------------------+---------+
| hyperparameter    |   value |
|-------------------+---------|
| gnn_dim_hidden    |     256 |
| dim_embedding     |      64 |
| dim_embedding_mlp |     128 |
| n_layers          |       1 |
+-------------------+---------+
```

The following are the results obtained.

```shell
+-----------+-------+
| metric    | score |
|-----------+-------|
| accuracy  | 0.736 |
| precision | 0.740 |
| recall    | 0.736 |
| f1-score  | 0.732 |
+-----------+-------+
```

The following is the report classification

```shell
                 precision    recall  f1-score   support

   Bunyavirales      0.815     0.775     0.794      1666
Mononegavirales      0.657     0.747     0.699      1322
    Nidovirales      0.709     0.883     0.787       901
   Ortervirales      0.716     0.804     0.758       945
 Picornavirales      0.740     0.497     0.595      1239
    Tymovirales      0.778     0.754     0.766       989

       accuracy                          0.736      7062
      macro avg      0.736     0.743     0.733      7062
   weighted avg      0.740     0.736     0.732      7062

```

## How to reproduce the experiment

First, it is necessary to download exactly the same dataset we used. Below are the commands for downloading our 
*training*, *validation* and *testing* dataset.

```shell
wget "https://drive.google.com/uc?export=download&id=1P_U6VBaEckH8Ycbg3CDZp2LABSreffBP" -O dataset.tar.gz
tar -xzvf dataset.tar.gz
```

Next, n order to reproduce the realized experiment, it is necessary to download the pre-trained model and run the ```main.py``` 
script again with the correct hyperparameters.

The following are the commands for properly downloading the pre-trained model.

```shell
wget "https://drive.google.com/uc?export=download&id=1PXJWIC8u7Pqy-V-lEC-itEjwDn3O7dFV" -O model.tar.gz
tar -xzvf model.tar.gz
```

With that done, we have nothing left but to run the ```main.py``` script as follows.

```shell
python3 main.py -read 250 \
                -overlap 200 \
                -k 14 \
                -model diff_pool \
                -hidden 256 \
                -embedding 64 \
                -embedding_mlp 128 \
                -layers 1
```

# Citations of works used

This section provides citations of all the work used in the development of this project.

* **UGformer**: [Nguyen, Dai Quoc, Tu Dinh Nguyen, and Dinh Phung. "Universal graph transformer self-attention networks." 
Companion Proceedings of the Web Conference 2022](https://arxiv.org/abs/1909.11855).
* **DiffPool**: [Ying, Zhitao, et al. "Hierarchical graph representation learning with differentiable pooling." 
Advances in neural information processing systems 31 (2018)](https://arxiv.org/abs/1806.08804).
