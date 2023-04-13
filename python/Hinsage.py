import pickle5 as pickle
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from copy import deepcopy

import stellargraph as sg
from stellargraph import StellarGraph
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import HinSAGELinkGenerator
from stellargraph.layer import HinSAGE, link_classification
from tensorflow.keras import Model, optimizers, losses, metrics
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from sklearn.model_selection import KFold

from utils import *

def main(args):

    # solve gpu issue
    fix_gpu()

    # load dataset
    wiki2csv = load_pickle(os.path.join(args.data_path,"wiki2csv"))
    csv2wiki = load_pickle(os.path.join(args.data_path,"csv2wiki"))
    embedding = load_pickle(os.path.join(args.emb_path,"Embedding"))
    num_csv = len(csv2wiki.keys())
    num_wiki = len(wiki2csv.keys())
    print(f"num_csv = {num_csv}, num_wiki = {num_wiki}")

    # make graph
    graph = make_graph(csv2wiki, embedding, num_csv, args)
    G_train, edge_ids_train, edge_labels_train, G_test, edge_ids_test, edge_labels_test = train_test_graph_split(graph)

    head_node_types = ["wiki","csv"]
    if edge_ids_train[0][0].find("csv") >= 0:
        head_node_types = ["csv", "wiki"]

    # make gen
    train_gen = make_generate(G_train, head_node_types, args) # HinSAGELinkGenerator
    test_gen = make_generate(G_test, head_node_types, args) # HinSAGELinkGenerator

    # test flow
    test_flow = test_gen.flow(edge_ids_test, edge_labels_test, shuffle = False) # 일종의 Dataloader?

    # K-Fold
    kfold = KFold(n_splits = args.num_folds,
                  shuffle = True)
    
    # result

    eval_results = {i : None for i in range(args.num_folds)}

    train_loss = deepcopy(eval_results)
    train_acc = deepcopy(eval_results)
    val_loss = deepcopy(eval_results)
    val_acc = deepcopy(eval_results)

    best_acc = 0
    best_model = None

    for idx, (train_index, val_index) in enumerate(kfold.split(edge_ids_train, edge_labels_train)):
        
        # model
        model = load_model(train_gen, args)  #print out: model.summary()        

        # check
        if idx == 0:
            print("Untrained model's Test Evaluation:")
            test(model, test_flow, args)

        train_flow = train_gen.flow(edge_ids_train[train_index], edge_labels_train[train_index], shuffle = True)
        val_flow = train_gen.flow(edge_ids_train[val_index], edge_labels_train[val_index], shuffle = False)
        
        # train
        history = model.fit(
                            train_flow,
                            validation_data = val_flow,
                            epochs = args.epochs,
                            verbose = 1,
                            shuffle = True, # batch shuffle: https://stackoverflow.com/questions/50184144/shuffle-in-the-model-fit-of-keras
                            use_multiprocessing = True,
                            workers = args.num_workers,
                        )
        result_one_fold = history.history

        train_loss[idx] = make_round(result_one_fold['loss'], 4)
        train_acc[idx] = make_round(result_one_fold['binary_accuracy'], 4)
        val_loss[idx] = make_round(result_one_fold['val_loss'], 4)
        val_acc[idx] = make_round(result_one_fold['val_binary_accuracy'], 4)

        # average of all epochs
        if np.mean(val_acc[idx]) > best_acc:
            best_acc = np.mean(val_acc[idx])
            best_model = model
            print(f"[Best model Changed] Best_acc = {best_acc}")

    # Report
    train_mean_loss = cal_mean(train_loss)
    train_mean_acc = cal_mean(train_acc)
    val_mean_loss = cal_mean(val_loss)
    val_mean_acc = cal_mean(val_acc)

    total_result = dict()
    for metric in ["train_mean_loss", "train_mean_acc", "val_mean_loss", "val_mean_acc", "train_loss", "train_acc", "val_loss", "val_acc"]:
        total_result[metric] = eval(metric)
    
    #print(f"\ntotal_result = {total_result}\n")

    # test
    print("Trained model's Test Evaluation:")
    test_result = test(best_model, test_flow, args)
    t_acc = round(test_result["binary_accuracy"],4)
    t_loss = round(test_result["loss"],4)


    # [save the all results] #
    folder_name = f"Acc_{t_acc}__Loss_{t_loss}"
    save_path = os.path.join(args.result_path, folder_name)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ## weight
    best_model.save(f"{save_path}/weight")

    ## figure
    save_figure(train_mean_loss, val_mean_loss, "Loss", save_path)
    save_figure(train_mean_acc, val_mean_acc, "Acc", save_path)

    ## result
    with open(f"{save_path}/total_result.pickle", "wb") as f:
        pickle.dump(total_result, f)

    ## datetime
    now_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(f"{save_path}/{now_time}.txt", "w") as f:
        f.write(str(args))
        f.write(f"\ntest_acc = {t_acc}")
        f.write(f"\ntest_loss = {t_loss}")

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path",
                        type = str,
                        default = "./Dataset",
                        help = "path of data")
    
    parser.add_argument("--emb_path",
                        type = str,
                        default = "./Result_for_embedding/Loss_1.2498",
                        help = "path of embedding")
    
    parser.add_argument("--result_path",
                        type = str,
                        default = "./Result_for_HinSage",
                        help = "path of data")

    parser.add_argument("--weight_toggle",
                        type = bool,
                        default = False,
                        help = "True: considering weight(frequency) between csv and wiki, False: otherwise it's not")

    parser.add_argument("--num_folds",
                        type = int,
                        default = 5,
                        help = "the number of K-fold splitted")
    
    parser.add_argument("--num_samples",
                        type = list,
                        default = [8, 4],
                        help = "the number of neighbors for each hop")
    
    parser.add_argument("--layer_sizes",
                        type = list,
                        default = [32, 16],
                        help = "the size of layer for each hop")   
    
    parser.add_argument("--batch_size",
                        type = int,
                        default = 200)
    
    parser.add_argument("--epochs",
                        type = int,
                        default = 100)
    
    parser.add_argument("--lr",
                        type = float,
                        default = 0.005)
    
    parser.add_argument("--drop_out",
                        type = float,
                        default = 0.3)
    
    parser.add_argument("--num_workers",
                        type = int,
                        default = -1)

    args = parser.parse_args()

    print("[Start] Link Predicition by using HinSage")
    print(f"args = {args}")

    main(args)

    print(f"args = {args}")
    print("[Done] Link Predicition by using HinSage")