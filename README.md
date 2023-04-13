# Upitt_RS_Project

  ### GOAL: How to recommend contents to students given context
  
  ### Assume: contents = Wikipedia or Youtube // Context = textbook pdf

  ### Approach: (1) Create a graph (2) Get a embedding for each node (3) link prediction
  
  ### Methods
    (1) Create a graph: WikiAPI + Keyword Network + Sentence Simiarity Network
    (2) Get Embedding: Deepwalk or MetaPath2Vec
    (3) Link Predcition: Heterogenous GraphSage




### [Jupyter Notebook]
```
#### 1. create a graph
* Preprocess.ipynb (make a csv file to get all of information, csv_keywords_df.csv)
* Wikipedia.ipynb (make four csv file to link csv and wiki, (csv_dict.pickle, wiki_dict.pickle, csv2wiki.pickle, wiki2csv.pickle))

#### 2. get a embedding for each node
* MetaPath2Vec.ipynb (make a graph and get embedding of each node, (Embedding, csv_wiki_graph))

#### 3. link prediction
* weighted_link_prediction.ipynb
```

#### Note
We save all files as pickle <br>
We use libraries like Pytorch, tensorflow, dgl, stellargraph <br>
[Link Predcition: Heterogenous GraphSage](https://stellargraph.readthedocs.io/en/stable/demos/link-prediction/hinsage-link-prediction.html, "Link_Prediction")

