from typing import Final
from typing import List
from typing import Dict

from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from glob import glob
import numpy as np
import logging
import random
import pickle
import time
import sys
import os

from Bio import SeqIO
from Bio import Entrez
from urllib.error import HTTPError
from urllib.error import URLError
import xmltodict

import pandas as pd
import pytaxonkit
from Bio.SeqRecord import SeqRecord

from utils import SEPARATOR
from utils import setup_logger

DATASET_PATH: Final = os.path.join(os.getcwd(), 'data', 'raw')
DICT_DATASET_PATH: Final = os.path.join(DATASET_PATH, 'dataset.pickle')
DF_DATASET_PATH: Final = os.path.join(DATASET_PATH, 'dataset.csv')


def split_fasta_file_on_processes(fasta_files: List[str], n_proc: int) -> List[List[str]]:
    n_files: int = len(fasta_files)
    n_files_for_process: int = n_files // n_proc
    rest: int = n_files % n_proc

    fasta_files_for_each_process: List[List[str]] = []
    rest_added: int = 0
    for i in range(n_proc):
        start: int = i * n_files_for_process + rest_added
        if rest > i:
            end: int = start + n_files_for_process + 1
            fasta_files_for_each_process.append(fasta_files[start:end])
            rest_added += 1
        else:
            end: int = start + n_files_for_process
            fasta_files_for_each_process.append(fasta_files[start:end])

    return fasta_files_for_each_process


def split_dataset_on_processes(
        fasta_files_for_each_process: List[List[str]],
        dataset: Dict[str, Dict[str, str]]) -> List[Dict[str, Dict[str, str]]]:
    # init list of n dicts for n process
    dataset_for_each_process: List[Dict[str, Dict[str, str]]] = []
    # for each files for process i
    for fasta_files_path in fasta_files_for_each_process:
        dataset_for_process_i: Dict[str, Dict[str, str]] = {}
        # for each file
        for fasta_file_path in fasta_files_path:
            # delegate file to process i
            dataset_for_process_i[fasta_file_path] = dataset[fasta_file_path]
        dataset_for_each_process.append(dataset_for_process_i)

    return dataset_for_each_process


def extract_tax_id(fasta_files_path: List[str], logger: logging.Logger) -> Dict[str, Dict[str, str]]:
    # init dataset
    dataset: Dict[str, Dict[str, str]] = {}
    # for each fasta file
    for fasta_file_path in fasta_files_path:
        # map all id in tax id
        mapping: Dict[str, str] = {}
        # read fasta file
        fasta_file = SeqIO.parse(open(fasta_file_path), 'fasta')
        # for each read in file
        for record in fasta_file:
            logger.info(f'Request for id: {record.id}')
            status_code: int = 0
            handle = None
            while status_code != 200:
                try:
                    # extract tax id
                    handle = Entrez.efetch(
                        db="nuccore",
                        id=record.id,
                        retmode="xml",
                        rettype="fasta"
                    )
                    status_code = handle.getcode()
                except HTTPError:
                    pass
                except URLError:
                    status_code = 200
                    logger.info(f'url error: skip {fasta_file_path}')
            if handle is None:
                continue
            response_dict = xmltodict.parse(handle)
            tax_id = response_dict['TSeqSet']['TSeq']['TSeq_taxid']
            # save id -> tax_id information
            mapping[record.id] = tax_id
            handle.close()
        # save all result in dataset
        dataset[fasta_file_path] = mapping
        # log file completed
        logger.info(f'{fasta_file_path} computed')

    return dataset


def extract_lineage(dataset: Dict[str, Dict[str, str]], logger: logging.Logger) -> pd.DataFrame:
    # init local dataframe
    df = pd.DataFrame()
    for fasta_file_path in dataset.keys():
        for seq_id in dataset[fasta_file_path].keys():
            # get tax_id
            tax_id = dataset[fasta_file_path][seq_id]
            # get lineage by tax_id
            df_result = pytaxonkit.lineage([tax_id])
            # get value of lineage
            try:
                full_lineage_ranks: List[str] = df_result['FullLineageRanks'][0].split(';')
                full_lineage: List[str] = df_result['FullLineage'][0].split(';')
                # create row of result
                values = {'File Path': fasta_file_path, 'ID': seq_id, 'TaxID': tax_id}
                values.update(dict(zip(full_lineage_ranks, full_lineage)))
                # merge result in local dataframe
                df = pd.concat([df, pd.DataFrame([values])])
            except AttributeError:
                continue
        # log file completed
        logger.info(f'{fasta_file_path} computed')

    return df


def generate_dataframe(len_read: int,
                       len_overlap: int,
                       dataset: pd.DataFrame,
                       dataset_status,
                       taxonomy_level: str,
                       indexes: List[int],
                       cluster: bool = False) -> pd.DataFrame:
    for index in tqdm(indexes,
                      total=len(indexes),
                      desc=f'Group all sequence by taxonomy level'):
        entry = dataset.loc[index]
        id_sequence: str = entry['ID']
        taxonomy_value: str = entry[taxonomy_level]
        fasta_file = SeqIO.parse(open(entry['File Path']), 'fasta')
        for sequence_fasta in fasta_file:
            if sequence_fasta.id == id_sequence:
                record = SeqRecord(
                    sequence_fasta.seq,
                    id=f'{id_sequence}_{taxonomy_value}',
                )
                with open(os.path.join(DATASET_PATH, f'{taxonomy_value}.fasta'), 'a') as output_handle:
                    SeqIO.write(record, output_handle, 'fasta')
                break

    for taxonomy_value in tqdm(dataset_status.keys(),
                               total=len(dataset_status.keys()),
                               desc=f'Generating {len_read} bp read with overlap size {len_overlap}...'):
        tmp_fasta_path: str = os.path.join(DATASET_PATH, f'{taxonomy_value}.fasta')
        gt_fasta_path: str = os.path.join(DATASET_PATH, f'{taxonomy_value}_gt.fasta')
        if len_overlap > 0:
            command: str = f'gt shredder ' \
                           f'-minlength {len_read} ' \
                           f'-maxlength {len_read} ' \
                           f'-overlap {len_overlap} ' \
                           f'-clipdesc yes ' \
                           f'{tmp_fasta_path} >> {gt_fasta_path}'
        else:
            command: str = f'gt shredder ' \
                           f'-minlength {len_read} ' \
                           f'-maxlength {len_read} ' \
                           f'-clipdesc yes ' \
                           f'{tmp_fasta_path} >> {gt_fasta_path}'
        os.system(command)
        # remove generated files
        for file_ext in ['', '.sds', '.ois', '.md5', '.esq', '.des', '.ssp']:
            os.system(f'rm {tmp_fasta_path}{file_ext}')

    # apply under-sampling with clustering
    cluster_path: str = os.path.join(DATASET_PATH, 'cluster')
    columns: Final = ['id_gene', taxonomy_level, 'start', 'end', 'sequence']
    df: pd.DataFrame = pd.DataFrame(columns=columns)
    for taxonomy_value in tqdm(dataset_status.keys(),
                               total=len(dataset_status.keys()),
                               desc="Merging of all reads into a single dataset..."):
        gt_fasta_path: str = os.path.join(DATASET_PATH, f'{taxonomy_value}_gt.fasta')
        if cluster:
            os.system(f'cd-hit-est '
                      f'-T 0 '
                      f'-d 0 '
                      f'-i {gt_fasta_path} '
                      f'-o {cluster_path} '
                      f'> /dev/null')
            os.system(f'rm {gt_fasta_path}')
            fasta_file = SeqIO.parse(open(cluster_path), 'fasta')
        else:
            fasta_file = SeqIO.parse(open(gt_fasta_path), 'fasta')

        for sequence_fasta in fasta_file:
            values: str = sequence_fasta.id
            values: List = values.split('_')
            id_gene: str = f'{values[0]}_{values[1]}'
            taxonomy_value: str = values[2]
            start: int = int(values[3])
            end: int = start + int(values[4])
            sequence = sequence_fasta.seq
            row_df: pd.DataFrame = pd.DataFrame(
                [[
                    id_gene,
                    taxonomy_value,
                    start,
                    end,
                    sequence
                ]],
                columns=columns
            )
            df = pd.concat([df, row_df])
        if cluster:
            os.system(f'rm {cluster_path} {cluster_path}.clstr')
        else:
            os.system(f'rm {gt_fasta_path}')

    return df.sample(frac=1)


def pre_processing(
        taxonomy_level: str,
        len_read: int = 250,
        len_overlap: int = 200,
        train_size: float = 0.6):
    # setup logger
    logger = setup_logger('logger', os.path.join(os.getcwd(), 'data/logger.log'))
    # get all fasta files path
    fasta_files: List[str] = glob(os.path.join(DATASET_PATH, '*.fasta'))
    # global dict of dataset
    dataset: Dict[str, Dict[str, str]] = {}
    # if tax id phase is completed, skip it
    if os.path.exists(DICT_DATASET_PATH):
        # load dataset with pickle
        with open(DICT_DATASET_PATH, 'rb') as handle:
            dataset = pickle.load(handle)
    else:
        # log start phase
        logger.info('Extract tax id phase')
        logger.info(SEPARATOR)
        # config Entrez email
        Entrez.email = sys.argv[1]
        # split fasta files on cpus
        fasta_files_for_each_process: List[List[str]] = split_fasta_file_on_processes(fasta_files, os.cpu_count())
        # create a process pool that uses all cpus
        start = time.time()
        with Pool(os.cpu_count()) as pool:
            results = pool.imap(partial(extract_tax_id, logger=logger), fasta_files_for_each_process)
            for local_dataset in results:
                dataset.update(local_dataset)
        # log finish phase
        logger.info(f'\nPhase completed in {(time.time()) - start}')
        logger.info(SEPARATOR)

    # save dataset with pickle
    with open(DICT_DATASET_PATH, 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if not os.path.exists(DF_DATASET_PATH):
        # init number of process
        n_proc: int = 5
        # log start phase
        logger.info('Extract taxonomy levels phase')
        logger.info(SEPARATOR)
        start = time.time()
        # split dataset for each process
        # split fasta files on cpus
        fasta_files_for_each_process: List[List[str]] = split_fasta_file_on_processes(fasta_files, n_proc)
        dataset_for_each_process: List[Dict[str, Dict[str, str]]] = split_dataset_on_processes(
            fasta_files_for_each_process,
            dataset
        )
        # create global df and split work on workers
        dataset_csv = pd.DataFrame()
        with Pool(n_proc) as pool:
            results = pool.imap(partial(extract_lineage, logger=logger), dataset_for_each_process)
            # merge each local dataset
            for local_dataset in results:
                dataset_csv = pd.concat([dataset_csv, local_dataset])
        # save global df
        dataset_csv.to_csv(DF_DATASET_PATH, index=False)
        # log finish phase
        logger.info(f'\nPhase completed in {(time.time()) - start}')
        logger.info(SEPARATOR)

    #  load dataset with pickle
    dataset_csv: pd.DataFrame = pd.read_csv(DF_DATASET_PATH)

    # group data by taxonomy_level
    dataset_csv = dataset_csv[dataset_csv[taxonomy_level].notnull()]
    dataset_status = dataset_csv.groupby(taxonomy_level)[taxonomy_level].count()

    # split idx in train, val and test set
    train_idx = []
    val_idx = []
    test_idx = []
    for label in dataset_status.keys():
        idx: List[int] = dataset_csv.index[dataset_csv[taxonomy_level] == label].values
        random.shuffle(idx)
        len_train = int(len(idx) * train_size)
        len_val = int((len(idx) - len_train) / 2)
        # split idx
        train_idx = np.append(train_idx, idx[:len_train])
        val_idx = np.append(val_idx, idx[len_train:len_train + len_val])
        test_idx = np.append(test_idx, idx[len_train + len_val:])

    # shuffle train, test and val
    random.shuffle(train_idx)
    random.shuffle(val_idx)
    random.shuffle(test_idx)

    # generate train dataframe
    logger.info(f'Generate training set...')
    train_df: pd.DataFrame = generate_dataframe(
        len_read=len_read,
        len_overlap=len_overlap,
        dataset=dataset_csv,
        dataset_status=dataset_status,
        taxonomy_level=taxonomy_level,
        indexes=train_idx,
        cluster=True
    )
    train_dataset_path = os.path.join(DATASET_PATH, f'{taxonomy_level}_train.csv')
    train_df.to_csv(train_dataset_path, index=False)
    logger.info(f'{train_dataset_path} generated!')

    # generate val set
    logger.info(f'Generate validation set...')
    val_df: pd.DataFrame = generate_dataframe(
        len_read=len_read,
        len_overlap=0,
        dataset=dataset_csv,
        dataset_status=dataset_status,
        taxonomy_level=taxonomy_level,
        indexes=val_idx,
        cluster=False
    )
    val_dataset_path: Final = os.path.join(DATASET_PATH, f'{taxonomy_level}_val.csv')
    val_df.to_csv(val_dataset_path, index=False)
    logger.info(f'{val_dataset_path} generated!')

    # generate test set
    logger.info(f'Generate testing set...')
    test_df: pd.DataFrame = generate_dataframe(
        len_read=len_read,
        len_overlap=0,
        dataset=dataset_csv,
        dataset_status=dataset_status,
        taxonomy_level=taxonomy_level,
        indexes=test_idx,
        cluster=False
    )
    test_dataset_path: Final = os.path.join(DATASET_PATH, f'{taxonomy_level}_test.csv')
    test_df.to_csv(test_dataset_path, index=False)
    logger.info(f'{test_dataset_path} generated!')


if __name__ == '__main__':
    pre_processing(taxonomy_level='order')
