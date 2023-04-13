# Upitt_RS_Project

  ## GOAL
  <b> How to recommend contents to students given context </b>
  
  ### Assume
  contents : Wikipedia or Youtube <br>
  Context : textbook pdf

  ### Approach
  (1) Create a graph (2) Get a embedding for each node (3) link prediction
  
  ### Methods
    (1) Create a graph: WikiAPI + Keyword Network + Sentence Simiarity Network
    (2) Get Embedding: Deepwalk or MetaPath2Vec
    (3) Link Predcition: Heterogenous GraphSage

  ## Codes
  ### [Python]
  Before you proceed it, you should set up two conda environments by using _environment_make_graph.yml_ and _environment_hinsage.yml_.

  ```
  1. create a graph 
  Wikipedia.py (environment_make_graph.yml must be set up)

  2. get a embedding for each node 
  make_embedding.py (environment_make_graph.yml must be set up)

  3. link prediction
  Hinsage.py (environment_hinsage.yml must be set up)

  ```

  ### [Jupyter Notebook]
  ```
  1. create a graph
  Preprocess.ipynb (make a csv file to get all of information, csv_keywords_df.csv)
  Wikipedia.ipynb (make four csv file to link csv and wiki, (csv_dict.pickle, wiki_dict.pickle, csv2wiki.pickle, wiki2csv.pickle))

  2. get a embedding for each node 
  MetaPath2Vec.ipynb (make a graph and get embedding of each node, (Embedding, csv_wiki_graph))

  3. link prediction
  weighted_link_prediction.ipynb
  ```

  ### [Note]
  We save all files as pickle <br>
  We use libraries like Pytorch, tensorflow, dgl, stellargraph <br>
  Document: [Link Predcition: Heterogenous GraphSage](https://stellargraph.readthedocs.io/en/stable/demos/link-prediction/hinsage-link-prediction.html, "Link_Prediction")

