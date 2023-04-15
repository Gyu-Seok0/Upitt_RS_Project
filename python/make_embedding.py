import os
import pickle
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime

import torch
from torch.optim import SparseAdam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import dgl
from dgl.nn.pytorch import MetaPath2Vec

import gc

def get_gpu():
    gc.collect()
    torch.cuda.empty_cache()

def load_pickle(name:str):
    with open(f'{name}.pickle','rb') as f:
        return pickle.load(f)

def make_node(dic : dict):
    src_node = []
    dst_node = []

    for src in dic.keys():
        for dst in dic[src]:
            src_node.append(src)
            dst_node.append(dst)

    return src_node, dst_node

def train(epochs, dataloader, model, optimizer, device, scheduler = None):
    
    total_loss = []
    min_loss = 1e10
    length = len(dataloader)

    for epoch in range(epochs):
        print(f"[Epoch:{epoch + 1:04d}]", end = " ")      
        epoch_loss = 0

        for idx, (pos_u, pos_v, neg_v) in enumerate(dataloader):

            pos_u = pos_u.to(device)
            pos_v = pos_v.to(device)
            neg_v = neg_v.to(device)

            # forward
            loss = model(pos_u, pos_v, neg_v)
            epoch_loss += loss.item()

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step(epoch + (idx / length))
                print(f"lr = {scheduler.get_last_lr()[0]}", end = " ")
        
        epoch_loss /= length
        min_loss = min(min_loss, epoch_loss)
        total_loss.append(epoch_loss)

        print(f"Loss = {epoch_loss:.4f}\n")

    return total_loss, min_loss


def save_figure(data, metric, result_path):

    plt.xlabel('Epochs', fontsize = 13)
    plt.ylabel(metric, fontsize = 13)

    splot = sns.lineplot(data)
    sfig = splot.get_figure()
    sfig.savefig(f'{result_path}/{metric}.png')
    
    plt.close(sfig)


def get_logger(logger_path):
    #logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # file
    file_handler = logging.FileHandler(f'{logger_path}.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def main(args):
    #logger -> pass

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    # load dataset
    csv2wiki = load_pickle(f"{args.data_path}/csv2wiki")
    wiki2csv = load_pickle(f"{args.data_path}/wiki2csv")

    num_csv = len(csv2wiki.keys()) # 56
    num_wiki = len(wiki2csv.keys()) # 2340

    # make nodes
    csv_src, wiki_dst = make_node(csv2wiki) # 7452
    wiki_src, csv_dst = make_node(wiki2csv) # 

    # make graph
    data_dict = {("csv", "csv2wiki", "wiki"): (csv_src, wiki_dst),
                 ("wiki", "wiki2csv", "csv") : (wiki_src, csv_dst)}
    
    graph = dgl.heterograph(data_dict)
    print(f"graph = {graph}")

    # model
    MetaPath = ["csv2wiki", "wiki2csv"] * (args.rwl // 2)

    model = MetaPath2Vec(graph,
                         MetaPath,
                         window_size = args.window_size,
                         negative_size = args.num_negative)
    print(f"model = {model}")
    model = model.to(device)

    # dataloader
    dataloader = DataLoader(torch.arange(graph.num_nodes("csv")),
                            batch_size = args.batch_size,
                            shuffle = True,
                            collate_fn = model.sample)

    # optimizer
    optimizer = SparseAdam(model.parameters(),
                           lr = args.lr)
    if args.scheduler:
        scheduler = CosineAnnealingWarmRestarts(optimizer,
                                                T_0 = 10,
                                                T_mult = 2,
                                                eta_min = 1e-5)
    else:
        scheduler = None

    # train
    total_loss, min_loss = train(args.epochs, dataloader, model, optimizer, device, scheduler)

    # result
    result_path = os.path.join(args.result_path, f"Loss_{min_loss:.4f}")

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    save_figure(data = total_loss,
                metric =  "Loss_for_Meta2Vec",
                result_path = result_path)
    
    ## result
    Embedding = model.node_embed.weight.detach().cpu().numpy()
    with open(f"{result_path}/Embedding.pickle", "wb") as f:
        pickle.dump(Embedding, f, pickle.HIGHEST_PROTOCOL)

    ## datetime
    now_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(f"{result_path}/{now_time}.txt", "w") as f:
        f.write(str(args))
        f.write(f"\n num_csv = {num_csv}")
        f.write(f"\n num_wiki = {num_wiki}")
        f.write(f"\n min_loss = {min_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path",
                        type = str,
                        default = "./Dataset/Total",
                        help = "the path of dataset")
    
    parser.add_argument("--result_path",
                        type = str,
                        default = "./Result_for_embedding/Total",
                        help = "the path of Result_for_embedding")
    
    parser.add_argument("--rwl",
                        type = int,
                        default = 40,
                        help = "the length of random walks")

    parser.add_argument("--window_size",
                        type = int,
                        default = 5,
                        help = "the length of window for positive")
    
    parser.add_argument("--num_negative",
                        type = int,
                        default = 3,
                        help = "the number of negatvie samples")

    parser.add_argument("--batch_size",
                        type = int,
                        default = 128,
                        help = "the number of dataset in one batch")

    parser.add_argument("--lr",
                        type = float,
                        default = 0.02,
                        help = "learning rate")
    
    parser.add_argument("--scheduler",
                        type = int,
                        default = 1,
                        help = "True of False")
    
    parser.add_argument("--epochs",
                        type = int,
                        default = 1000)

    args = parser.parse_args()
    
    print(args)
    print("[Start] Making Embedding")

    main(args)

    print("[Done] Making Embedding")
    print(args)






