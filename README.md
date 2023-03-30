# Upitt_RS_Project


## GOAL: How to recommend matetrials to students given context(e.g., textbook pdf) </b>

## Process: (1) create a graph -> (2) get a embedding for each node -> (3) link prediction </b>

### 1. create a graph
- Preprocess.ipynb (make a csv file to get all of information, csv_keywords_df.csv)
- Wikipedia.ipynb (make four csv file to link csv and wiki, (csv_dict.pickle, wiki_dict.pickle, csv2wiki.pickle, wiki2csv.pickle))

### 2. get a embedding for each node
- MetaPath2Vec.ipynb (make a graph and get embedding of each node, (Embedding, csv_wiki_graph))

### 3. link prediction
- TBA

### Note
- We save all files as pickle
- We use libraries like Pytorch, dgl.