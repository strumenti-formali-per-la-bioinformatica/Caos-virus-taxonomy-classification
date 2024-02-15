# Table of Contents

* [Overview](#overview)
* [Requirements](#requirements)
  * [Dataset](#dataset)
* [Usage](#usage)
* [Results](#results)
  * [How to run the code](#how-to-run-the-code)
* [Citations of works used](#citations-of-works-used)

# Overview
A taxonomy (or taxonomical classification) is a scheme of classification, especially a hierarchical classification, in which things are organized into groups or types.
In the world of biology being able to correctly identify the taxonomy of something can give us precious informations about it.\
This project, inspired and made possible by the previous work of our colleagues A.Cirillo and N.Gagliarde, explores the possibility
of using overlap and chaos graphs to train a GNN that is able to correctly identify the taxonomy of a virus.

# Requirements
This is section is shared with and adapted from original project of A.Cirillo and N.Gagliarde.

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
pip install complexcgr
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
* ```model```: select the model to use. In the current version only a diff_pool model is supported.
* ```hidden_size```: defines the number of neurons to be used in the hidden layers.
* ```n_layers```: defines the number of layers.
* ```embedding```: defines the size of the embeddings. 
* ```embedding_mlp```: defines the size of the embeddings in the fully connected layers. 

# Results

This section shows the results obtained.

The experiments were carried out by setting the read size to 250bp with overlap of 200bp.

Three different models, more specifically **UGformer** and **DiffPool**, were tested during the various 
experiments, only the code for the DiffPool model is made publicly available as it is the one that performed best. References to these two models can be found in the "*[Citations of works used](#citations-of-works-used)*" 
section. 

In addition to testing the different hyperparameters, the size of the kmer, ```len_kmer``` parameter, was also tested, 
varying it from a size of 4 to a maximum of 16 for the overlap graph and from a size of 4 to a maximum size of 9 for the chaos graph.
The best results were  obtained with a kmer length for the former and of 9 for tha latter. More experiments will be conducted in the near future to establish how the model behaves with larger kmer lengths.

## Chaos graph results

```shell
+-----------+-------+
| metric    | score |
|-----------+-------|
| accuracy  | 0.736 |
| precision | 0.730 |
| recall    | 0.730 |
| f1-score  | 0.726 |
+-----------+-------+
```

The following is the report classification

```shell
                 precision    recall  f1-score   support

   Bunyavirales      0.742     0.863     0.798      1747
Mononegavirales      0.689     0.633     0.660      1258
    Nidovirales      0.731     0.868     0.794      1135
   Ortervirales      0.770     0.618     0.686       849
 Picornavirales      0.646     0.587     0.615      1244
    Tymovirales      0.827     0.737     0.780      1024

       accuracy                          0.730      7257
      macro avg      0.734     0.718     0.722      7257
   weighted avg      0.730     0.730     0.726      7257

```

## Overlap graph results

```shell
+-----------+-------+
| metric    | score |
|-----------+-------|
| accuracy  | 0.804 |
| precision | 0.804 |
| recall    | 0.799 |
| f1-score  | 0.801 |
+-----------+-------+
```

The following is the report classification

```shell
                 precision    recall  f1-score   support

   Bunyavirales      0.858     0.843     0.850      1747
Mononegavirales      0.742     0.764     0.753      1258
    Nidovirales      0.875     0.846     0.860      1135
   Ortervirales      0.748     0.806     0.776       849
 Picornavirales      0.693     0.744     0.717      1244
    Tymovirales      0.889     0.779     0.830      1024

       accuracy                          0.799      7257
      macro avg      0.801     0.797     0.798      7257
   weighted avg      0.804     0.799     0.801      7257

```


## How to run the code

Assuming a correctly installed environment and that the user followed all the preprocessing steps, we have nothing left but to run the ```main.py``` script as follows.

```shell
python3 main.py -read 250 \
                -overlap 200 \
                -k 12 \
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
* **virus-taxonomy-classification**: By A.Cirillo and N.Gagliarde.
