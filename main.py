from typing import Optional
from typing import Dict
from typing import Any

import numpy as np
import argparse
import logging

import torch
import os

from utils import SEPARATOR
from utils import create_test_name
from utils import test_check
from utils import create_folders
from utils import setup_logger
from utils import save_result
from utils import close_loggers

from dna.dataset import DNADataset
from torch_geometric.loader import DataLoader

from sklearn.utils import class_weight
from torch.optim import AdamW

import models
from models.diff_pool import DiffPool
from models import train
from models import predict

from sklearn.metrics import classification_report


def main(
        len_read: int,
        len_overlap: int,
        k_size: int,
        model_selected: str,
        hyperparameter: Dict[str, Any],
        batch_size: int
):
    # generate test name
    test_name: str = create_test_name(
        len_read=len_read,
        len_overlap=len_overlap,
        k_size=k_size,
        hyperparameter=hyperparameter
    )
    # check if this configuration is already tested
    if not test_check(model_name=model_selected, parent_name=test_name):
        # create folders and get path
        log_path, model_path = create_folders(model_name=model_selected, parent_name=test_name)

        # init loggers
        logger: logging.Logger = setup_logger('logger', os.path.join(log_path, 'logger.log'))
        train_logger: logging.Logger = setup_logger('train', os.path.join(log_path, 'train.log'))

        # init train and validation dataset
        train_dataset = DNADataset(
            root=os.path.join(os.getcwd(), 'data'),
            k_size=k_size,
            taxonomy_level='order',
            len_read=len_read,
            len_overlap=len_overlap
        )
        val_dataset = DNADataset(
            root=os.path.join(os.getcwd(), 'data'),
            k_size=k_size,
            taxonomy_level='order',
            len_read=len_read,
            len_overlap=len_overlap,
            dataset_type='val'
        )

        # add number of features node and number of classes
        hyperparameter['dim_features'] = train_dataset.num_node_features
        hyperparameter['dim_edge_features'] = train_dataset.num_edge_features
        hyperparameter['n_classes'] = train_dataset.num_classes

        # log information
        logger.info(f'Read len: {len_read}')
        logger.info(f'Overlap len: {len_overlap}')
        logger.info(f'Kmers size: {k_size}')
        logger.info(f'Batch size: {batch_size}')
        logger.info(f'Number of features node: {train_dataset.num_node_features}')
        logger.info(f'Number of features edge: {train_dataset.num_edge_features}')
        logger.info(f'Number of train graph: {len(train_dataset)}')
        logger.info(f'Number of val graph: {len(val_dataset)}')
        logger.info(f'Number of class: {train_dataset.num_classes}')
        logger.info(SEPARATOR)

        # create train, and validation data
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        # print dataset status
        logger.info('No. records train set')
        logger.info(train_dataset.dataset_status())
        logger.info('No. records val set')
        logger.info(val_dataset.dataset_status())

        # set device gpu if cuda is available
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # evaluate weights for loss function
        y = []
        for idx, label in enumerate(train_dataset.n_records_for_label):
            y = np.append(y, [idx] * label)
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
        # tensor([0.7169, 0.9000, 1.1258, 1.3580, 0.9952, 1.1567])
        class_weights: torch.Tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

        # init model
        model: Optional[models.Model] = None
        if model_selected == 'diff_pool':
            # get max nodes number in graph
            max_num_nodes: int = 0
            for i in range(len(train_dataset)):
                num_nodes: int = train_dataset[i].x.size()[0]
                if max_num_nodes < num_nodes:
                    max_num_nodes = num_nodes
            hyperparameter['max_num_nodes'] = max_num_nodes
            model: models.Model = DiffPool(
                hyperparameter=hyperparameter,
                weights=class_weights
            )
        if model_selected == 'ug_gat':
            model: models.Model = UGFormerGAT(
                hyperparameter=hyperparameter,
                weights=class_weights
            )
        if model_selected == 'ug_gcn':
            model: models.Model = UGFormerGCN(
                hyperparameter=hyperparameter,
                weights=class_weights
            )

        # log model hyper parameters
        logger.info('Model hyperparameter')
        logger.info(model.print_hyperparameter())

        # init optimizer
        optimizer = AdamW(model.parameters())
        # put model on gpu if it is available
        model.to(device)

        # train model
        train(
            model=model,
            train_loader=train_dataloader,
            optimizer=optimizer,
            model_path=model_path,
            device=device,
            epochs=1000,
            evaluation=True,
            val_loader=val_dataloader,
            patience=10,
            logger=train_logger
        )

        # close loggers
        close_loggers([train_logger, logger])
        del train_logger
        del logger

    # get path of model and log
    log_path, model_path = create_folders(model_name=model_selected, parent_name=test_name)
    # init loggers
    logger: logging.Logger = setup_logger('logger', os.path.join(log_path, 'logger.log'))

    # load test dataset
    test_dataset = DNADataset(
        root=os.path.join(os.getcwd(), 'data'),
        k_size=k_size,
        taxonomy_level='order',
        len_read=len_read,
        len_overlap=len_overlap,
        dataset_type='test'
    )

    # log test dataset status
    logger.info('No. records test set')
    logger.info(test_dataset.dataset_status())

    # load model
    model: models.Model = torch.load(os.path.join(model_path, 'model.h5'),map_location='cuda' if torch.cuda.is_available() else 'cpu')
    # set device gpu if cuda is available
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set model on gpu
    model.to(device)

    # create test data loader
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # test model
    y_true, y_pred = predict(
        model=model,
        test_loader=test_dataloader,
        device=device
    )

    # log classification report
    report: str = classification_report(
        y_true,
        y_pred,
        digits=3,
        zero_division=1,
        target_names=test_dataset.labels.keys()
    )
    logger.info(report)

    # close loggers
    close_loggers([logger])
    del logger

    # save result
    save_result(
        result_csv_path=os.path.join(os.getcwd(), 'log', model_selected, 'results.csv'),
        len_read=len_read,
        len_overlap=len_overlap,
        hyperparameter=model.hyperparameter,
        y_true=y_true,
        y_pred=y_pred
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-read', dest='len_read', action='store',
                        type=int, default=250, help='define length of reads')
    parser.add_argument('-overlap', dest='len_overlap', action='store',
                        type=int, default=200, help='define length of overlapping between training reads')
    parser.add_argument('-k', dest='k_size', action='store',
                        type=int, default=14, help='define length of kmers')
    parser.add_argument('-batch', dest='batch_size', action='store',
                        type=int, default=512, help='define batch size')
    parser.add_argument('-model', dest='model', action='store',
                        type=str, default='diff_pool', help='select the model to be used')
    parser.add_argument('-hidden', dest='hidden_size', action='store',
                        type=int, default=256, help='define number of hidden channels')
    parser.add_argument('-embedding', dest='embedding', action='store',
                        type=int, default=64, help='define embedding size')
    parser.add_argument('-embedding_mlp', dest='embedding_mlp', action='store',
                        type=int, default=128, help='define mlp embedding size')
    parser.add_argument('-layers', dest='n_layers', action='store',
                        type=int, default=1, help='define number of model layers')
    parser.add_argument('-att_layers', dest='n_att_layers', action='store',
                        type=int, default=1, help='define number of attention layers')
    parser.add_argument('-tf_heads', dest='tf_heads', action='store',
                        type=int, default=1, help='define number of transformer heads')
    parser.add_argument('-gat_heads', dest='gat_heads', action='store',
                        type=int, default=1, help='define number of gat heads')

    args = parser.parse_args()

    # check model selected
    if args.model not in ['diff_pool', 'ug_gat', 'ug_gcn']:
        raise Exception('select one of these models: ["diff_pool", "ug_gat", "ug_gcn"]')

    # create dict of model hyperparameter
    parameter: Dict[str, Any] = {}
    if args.model == 'diff_pool':
        parameter['hidden_size'] = args.hidden_size
        parameter['dim_embedding'] = args.embedding
        parameter['dim_embedding_mlp'] = args.embedding_mlp
        parameter['n_layers'] = args.n_layers
    if args.model == 'ug_gat':
        parameter['hidden_size'] = args.hidden_size
        parameter['dim_embedding'] = args.embedding
        parameter['n_layers'] = args.n_layers
        parameter['att_layers'] = args.n_att_layers
        parameter['tf_heads'] = args.tf_heads
        parameter['gat_heads'] = args.gat_heads
    if args.model == 'ug_gcn':
        parameter['hidden_size'] = args.hidden_size
        parameter['dim_embedding'] = args.embedding
        parameter['n_layers'] = args.n_layers
        parameter['att_layers'] = args.n_att_layers
        parameter['tf_heads'] = args.tf_heads

    main(
        len_read=args.len_read,
        len_overlap=args.len_overlap,
        k_size=args.k_size,
        model_selected=args.model,
        hyperparameter=parameter,
        batch_size=args.batch_size
    )
