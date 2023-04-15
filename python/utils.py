import pickle5 as pickle
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


def load_pickle(name : str):
    with open(f'{name}.pickle','rb') as f:
        return pickle.load(f)

def make_pairs(dict : dict) -> list:
    pairs = []
    for key, values in dict.items():
        for value in values:
            pairs.append((key, value))
    return pairs

def rename_for_value(column : str,
                     df : pd.DataFrame
                     ) -> pd.Series:
    return df[f"{column}"].apply(lambda x: f"{column}-{x}")

def make_graph(csv2wiki : dict, 
               embedding : np.array, 
               num_csv : int,
               args : argparse
               ) -> StellarGraph:
    
    # pairs
    pairs = make_pairs(csv2wiki)
    edge_df = pd.DataFrame(pairs, columns = ["csv", "wiki"])
    print(f"edge_df.shape = {edge_df.shape}")

    if not args.weight_toggle:
        edge_df = edge_df.drop_duplicates()
        print("\n[After applying drop_duplicate]")
        print(f"edge_df.shape = {edge_df.shape}")
    
    # rename
    for column in ["csv", "wiki"]:
        edge_df[column] = rename_for_value(column, edge_df)
    print(f"\n[After applying rename_for_value]")
    print(edge_df.head())

    # get emb
    emb_csv, emb_wiki = embedding[:num_csv, :], embedding[num_csv:, :]
    node_csv = pd.DataFrame(emb_csv).rename(index = lambda x : f"csv-{x}")
    node_wiki = pd.DataFrame(emb_wiki).rename(index = lambda x : f"wiki-{x}")

    print(f"node_csv = {node_csv.head()}")
    print(f"node_wiki = {node_wiki.head()}")

    # graph
    graph = StellarGraph(nodes = {"csv":node_csv, "wiki":node_wiki}, 
                         edges = edge_df,
                         source_column = "csv",
                         target_column = "wiki",
                        )
    print("\n[Graph]")
    print(graph.info())

    return graph

def train_test_graph_split(graph: StellarGraph) -> tuple:

    edge_splitter_test = EdgeSplitter(graph)
    G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
    p=0.02, method="global", keep_connected=True, edge_label = "default", 
    )

    edge_splitter_train = EdgeSplitter(G_test)
    G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
    p=0.021, method="global", keep_connected=True, edge_label = "default"
    )

    return (G_train, edge_ids_train, edge_labels_train,
            G_test,  edge_ids_test,  edge_labels_test)

def make_generate(graph, head_node_types, args):
    gen = HinSAGELinkGenerator(graph, 
                               args.batch_size,
                               args.num_samples,
                               head_node_types = head_node_types) # it can cause error
    return gen

def load_model(train_gen ,args):

    hinsage = HinSAGE(
                      layer_sizes = args.layer_sizes,
                      generator = train_gen,
                      bias = True,
                      dropout = args.drop_out
                      )
    
    assert len(args.layer_sizes) == len(args.num_samples) # check whether it matches the number between them.

    # head
    x_inp, x_out = hinsage.in_out_tensors()
    score_prediction = link_classification(edge_embedding_method = "ip")(x_out) # ip = inner proudct
    
    # extractor
    model = Model(inputs = x_inp,
                  outputs = score_prediction)
    
    # etc
    model.compile(
        optimizer=optimizers.Adam(lr = args.lr),
        loss = losses.binary_crossentropy,
        metrics = [metrics.binary_accuracy]
        )
    return model

def test(model, test_flow, args):
    test_metrics = model.evaluate(
                                    test_flow,
                                    verbose = 1,
                                    use_multiprocessing = True,
                                    workers = args.num_workers
                                )

    result = dict(zip(model.metrics_names, test_metrics))

    for name, val in result.items():
        print("\t{}: {:0.4f}".format(name, val))
    
    return result

def make_round(l : list, digit : int) -> list:
    return list(map(lambda x: round(x , digit), l))

def cal_mean(data : dict):
    return np.array(list(data.values())).mean(axis = 0)

def save_figure(train_data, val_data, metric, result_path):
    data = {"train" : train_data,
            "val" : val_data}
    df = pd.DataFrame(data)

    plt.xlabel('Epochs', fontsize = 13)
    plt.ylabel(metric, fontsize = 13)

    splot = sns.lineplot(data = df)
    sfig = splot.get_figure()
    sfig.savefig(f'{result_path}/{metric}.png')
    
    plt.close(sfig)