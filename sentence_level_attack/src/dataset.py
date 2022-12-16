# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import os

from datasets import load_dataset
from src.utils import target_offset

# def load_data(args):
def load_data(dataset_name):
    data_files = {
        'train': os.path.join('data', dataset_name, 'train_processed.json'),
        'validation': os.path.join('data', dataset_name, 'test_processed.json')
    }
    datasets = load_dataset('json', data_files=data_files)

    num_labels = 0
    if dataset_name == 'dbpedia14':
        num_labels = 14
    elif dataset_name == 'ag_news':
        num_labels = 4
    elif dataset_name == 'imdb':
        num_labels = 2
    elif dataset_name == 'yelp':
        num_labels = 2

    return datasets, num_labels
    # if args.dataset == "dbpedia14":
    #     dataset = load_dataset("csv", column_names=["label", "title", "sentence"],
    #                             data_files={"train": os.path.join(args.data_folder, "dbpedia_csv/train.csv"),
    #                                         "validation": os.path.join(args.data_folder, "dbpedia_csv/test.csv")})
    #     dataset = dataset.map(target_offset, batched=True)
    #     num_labels = 14
    # elif args.dataset == "ag_news":
    #     dataset = load_dataset("ag_news")
    #     num_labels = 4
    # elif args.dataset == "imdb":
    #     dataset = load_dataset("imdb", ignore_verifications=True)
    #     num_labels = 2
    # elif args.dataset == "yelp":
    #     dataset = load_dataset("yelp_polarity")
    #     num_labels = 2
    # elif args.dataset == "mnli":
    #     dataset = load_dataset("glue", "mnli")
    #     num_labels = 3
    # dataset = dataset.shuffle(seed=0)
    
    

    # return dataset, num_labels