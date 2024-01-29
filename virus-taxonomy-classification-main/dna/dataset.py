from typing import Union
from typing import Tuple
from typing import List
from typing import Dict

from tabulate import tabulate
from tqdm import tqdm
import pandas as pd
import torch
import os

from multiprocessing import Pool
from functools import partial

from torch_geometric.data import Data
from dna.chaos import ChaosGraph
from torch_geometric.data import Dataset
from torch_geometric.utils import from_networkx
from dna.de_bruijn import DeBruijnGraph
from dna.overlap import OverlapGraph


class DNADataset(Dataset):
    def __init__(self,
                 root: str,
                 k_size: int = 5,
                 taxonomy_level: str = 'order',
                 len_read: int = 250,
                 len_overlap: int = 200,
                 dataset_type: str = 'train',
                 transform=None,
                 pre_transform=None):

        self.k_size: int = k_size
        self.taxonomy_level: str = taxonomy_level
        self.len_read: int = len_read
        self.len_overlap: int = len_overlap
        self.dataset_type: str = dataset_type

        self.labels: Dict[str, int] = {}
        self.n_records_for_label = None
        self.n_graphs: int = 0
        self.df = None

        super(DNADataset, self).__init__(
            root,
            transform,
            pre_transform
        )

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        """ If this file exist in raw_dir, the download is not triggered. """
        return [os.path.join(self.raw_dir, f'{self.taxonomy_level}_{self.dataset_type}.csv')]

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        """ If these files are found in processed_dir, processing is skipped. """
        # read dataset
        self.df: pd.DataFrame = pd.read_csv(os.path.join(self.raw_dir, f'{self.taxonomy_level}_'
                                                                       f'{self.dataset_type}.csv'))
        # deletes all reads that have a length that differs by at most 10 from the established length
        self.df = self.df[self.df['sequence'].str.len() >= (self.len_read - 10)]
        self.df = self.df.reset_index(drop=True)
        # group and count by taxonomy level
        self.n_records_for_label = self.df.groupby(self.taxonomy_level)[self.taxonomy_level].count()
        # map label in integer
        for idx, label in enumerate(self.n_records_for_label.keys()):
            self.labels[label] = idx
        self.n_graphs = self.n_records_for_label.values.sum()

        # compute processed files list
        processed_files: List[str] = []
        for idx in range(self.n_graphs):
            if self.dataset_type == "train":
                processed_files.append(f'{self.len_read}_'
                                       f'{self.len_overlap}_'
                                       f'{self.taxonomy_level}_'
                                       f'{self.k_size}_'
                                       f'{idx}.pt')
            else:
                processed_files.append(f'{self.dataset_type}_'
                                       f'{self.len_read}_'
                                       f'{self.taxonomy_level}_'
                                       f'{self.k_size}_'
                                       f'{idx}.pt')

        return processed_files

    def download(self):
        raise Exception('please run pre_processing.py first')

    def process(self):
        # split sequence on different process
        n_reads: int = self.n_graphs
        n_proc: int = os.cpu_count()
        n_reads_for_process: int = n_reads // n_proc
        rest: int = n_reads % n_proc
        # create start and end index for all process
        start_end_idx_for_process: List[Tuple[int, int]] = []
        rest_added: int = 0
        for i in range(n_proc):
            start: int = i * n_reads_for_process + rest_added
            if rest > i:
                end: int = start + n_reads_for_process + 1
                start_end_idx_for_process.append((start, end))
                rest_added += 1
            else:
                end: int = start + n_reads_for_process
                start_end_idx_for_process.append((start, end))
        # call create_graph_from_sequence in concurrent
        with Pool(n_proc) as pool:
            pool.map(partial(self.create_graph_from_sequence), start_end_idx_for_process)

    def create_graph_from_sequence(self, start_end_idx: Tuple[int, int]):
        # read dataframe from start to end index
        for idx in tqdm(range(start_end_idx[0], start_end_idx[1]), total=len(start_end_idx)):
            # read sequence from dataset
            sequence: str = self.df.loc[idx, 'sequence']
            # generate de bruijn graph and convert it in geometric data
            graph = ChaosGraph(sequence, self.k_size)
            
            ptg = from_networkx(
                graph.graph_chaos,
                group_node_attrs=graph.node_attr,
            )
            ptg.y = torch.tensor([self.labels[self.df.loc[idx, self.taxonomy_level]]])
            # save geometric data
            if self.dataset_type == "train":
                file_path = f'{self.len_read}_' \
                            f'{self.len_overlap}_' \
                            f'{self.taxonomy_level}_' \
                            f'{self.k_size}_' \
                            f'{idx}.pt'
            else:
                file_path = f'{self.dataset_type}_' \
                            f'{self.len_read}_' \
                            f'{self.taxonomy_level}_' \
                            f'{self.k_size}_' \
                            f'{idx}.pt'
            # save ptg file
            torch.save(ptg, os.path.join(self.processed_dir, file_path))

    def len(self) -> int:
        """ Return number of graph """
        return self.n_graphs

    def get(self, idx: int) -> Data:
        """ Return the idx-th graph. """
        if self.dataset_type == "train":
            file_path = f'{self.len_read}_' \
                        f'{self.len_overlap}_' \
                        f'{self.taxonomy_level}_' \
                        f'{self.k_size}_' \
                        f'{idx}.pt'
        else:
            file_path = f'{self.dataset_type}_' \
                        f'{self.len_read}_' \
                        f'{self.taxonomy_level}_' \
                        f'{self.k_size}_' \
                        f'{idx}.pt'
        data = torch.load(os.path.join(self.processed_dir, file_path))

        return data

    @property
    def num_classes(self) -> int:
        return len(self.n_records_for_label.keys())

    def dataset_status(self):
        table: List[List[str, int]] = [[label, record] for label, record in self.n_records_for_label.items()]
        table_str: str = tabulate(
            tabular_data=table,
            headers=['label', 'no. records'],
            tablefmt='psql'
        )
        return f'\n{table_str}\n'

