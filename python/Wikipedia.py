import argparse
import pandas as pd
import ast
from mediawikiapi import MediaWikiAPI
mediawiki = MediaWikiAPI()

import pickle
from nltk.stem import PorterStemmer, LancasterStemmer,WordNetLemmatizer
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.nn import DataParallel
import torch
import gc
from keybert import KeyBERT
from tqdm import tqdm
import scann
from math import sqrt


def generate_embeddings(data_ls, model):
    embeddings = model.encode(data_ls)
    return embeddings
     

def get_sim_scores(csv_keyword, wiki_keywords, model):
    csv_emb_df =  pd.DataFrame(generate_embeddings([csv_keyword],model))
    wiki_embs_df = pd.DataFrame(generate_embeddings([wiki_keywords],model))
    k = int(sqrt(len(wiki_keywords)))
    searcher = scann.scann_ops_pybind.builder(wiki_embs_df, 10, "dot_product").tree(num_leaves=k, num_leaves_to_search=int(k/20), training_sample_size=2500).score_brute_force(2).reorder(7).build()

    matched_wiki_keyword_indices, wiki_sim_scores = searcher.search(csv_emb_df.loc[0])

    return np.mean(wiki_sim_scores) if len(matched_wiki_keyword_indices) > 0 else 0.0



def get_avg_scores(csv_keyword, wiki_keywords, model):
    get_gpu()
    with torch.no_grad():
        csv_emb = model.encode(csv_keyword)
        wiki_embs = model.encode(wiki_keywords)

    avg_score = util.pytorch_cos_sim(csv_emb, wiki_embs).mean().item()
    return avg_score

def get_keywords(sample,kw_model):

    keywords = kw_model.extract_keywords(sample, stop_words = 'english', use_maxsum = True, top_n = 15)

    return [k[0] for k in keywords]



def get_keywords_T5(sample, model, tokenizer, device):
    # preprocess
    task_prefix = "Keywords: "
    input_sequences = [task_prefix + sample]
    input_ids = tokenizer(
        input_sequences, return_tensors="pt", truncation=True
    ).input_ids.to(device)

    # generate
    get_gpu()
    with torch.no_grad():
        output = model.generate(input_ids, no_repeat_ngram_size=3, num_beams=4)
        predicted = tokenizer.decode(output[0], skip_special_tokens=True)

    return predicted

def get_id(key, mapping_id_dict):
    if key not in mapping_id_dict.keys():
        mapping_id_dict[key] = len(mapping_id_dict)
    return mapping_id_dict[key]

def create_connection(connect_dict, key, value):
    if key not in connect_dict.keys():
        connect_dict[key] = [value]
    else:
        connect_dict[key].append(value)

def save_wiki_metadata(wiki_metadata, title, keywords, url):
    if title not in wiki_metadata.keys():
        data = {
                "keywords" : keywords,
                "url" : url
                }
        wiki_metadata[title] = data
            
def cleaning(keywords, ps, lm):
    after_ps = set(map(lambda x: ps.stem(x), keywords))
    after_ps_lm = set(map(lambda x: lm.lemmatize(x), after_ps))
    return list(after_ps_lm)

def get_gpu():
    gc.collect()
    torch.cuda.empty_cache()

def main(args):

    # device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = "cpu"
    print(f"device = {device}")


    # load dataset
    ck_df = pd.read_csv(args.df_path)
    ck_df["keywords"] = ck_df['keywords'].apply(lambda x: ast.literal_eval(x))


    # keywords extraction model from wiki text
    if False: keyword_model = T5ForConditionalGeneration.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
    if False: keyword_tokenizer = T5Tokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    model1 = SentenceTransformer(args.model_name).to(device)
    kw_model = KeyBERT(model=model1)

    # threshold to avg_score
    avg_score_th = 0.4

    # word cleaning
    ps = PorterStemmer()
    lm = WordNetLemmatizer()

    # save 
    wiki_metadata = {} # key = title, value = metadata
    csv_metadata = {} # key = csv_files, value = metadata

    wiki_mapping_id = {} # key = title, value = id
    csv_mapping_id = {} # same above

    csv2wiki = {} # interaction from csv to wiki
    wiki2csv = {} # interaction from wiki to csv

    keyword_pairs = {}
    # iteration for csv files
    for csv in tqdm(ck_df.itertuples()):
        
        # cleaning
        clean_keyword = csv.keywords # cleaning(csv.keywords, ps, lm)
        
        # save metadata for csv 
        csv_file_name = csv.csv_file
        csv_metadata[csv_file_name] = clean_keyword
        print(f"[{csv_file_name}]")

        # csv_keyword
        for csv_keyword in clean_keyword:

            # wiki based on searching for csv_keyword
            wiki_list = mediawiki.search(csv_keyword)

            for wiki_candidate in wiki_list:

                # find page
                try:
                    wiki_page = mediawiki.page(wiki_candidate)
                    wiki_title = wiki_page.title
                    wiki_url = wiki_page.url
                    summary = wiki_page.summary
                    page_content = wiki_page.content
                except:
                    print("Cannot Find any page! Continue next")
                    continue


                # wiki_keywords
                if wiki_title not in wiki_metadata.keys():
                    wiki_keywords = get_keywords(page_content, kw_model)
                    if False: wiki_keywords = wiki_keywords.split(",")
                else:
                    wiki_keywords = wiki_metadata[wiki_title]["keywords"]

                # get_avg scores
                if not(len(wiki_keywords) == 0):
                    try:
                        avg_score = keyword_pairs[(csv_keyword, tuple(wiki_keywords))]
                    except:
                        avg_score = get_avg_scores(csv_keyword, wiki_keywords, model1)
                        keyword_pairs[(csv_keyword, tuple(wiki_keywords))] = avg_score

                    # save
                    if avg_score > avg_score_th:
                        
                        # get id
                        wiki_id = get_id(wiki_title, wiki_mapping_id)
                        csv_id = get_id(csv_file_name, csv_mapping_id)

                        # create connection
                        create_connection(wiki2csv, wiki_id, csv_id)
                        create_connection(csv2wiki, csv_id, wiki_id)

                        # save wiki_metadata
                        save_wiki_metadata(wiki_metadata, wiki_title, wiki_keywords, wiki_url)

                        print(f"[Connect] csv_keyword = {csv_keyword}, wiki_title = {wiki_title}, avg_score = {round(avg_score,2)}")

        # save the all files as pickle

        names = ["wiki_metadata","csv_metadata","wiki_mapping_id","csv_mapping_id","csv2wiki","wiki2csv"]
        for name in names:
            with open(f'{args.out_path}/{name}.pickle','wb') as f:
                pickle.dump(eval(name), f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--df_path", type = str, default = "./Dataset/Stanford_big/test.csv", help = "the path of csv2keywords")
    parser.add_argument("--out_path", type=str, default="./Dataset/Stanford_big/", help="output folder for pickle files")
    parser.add_argument("--model_name",type=str,default="all-MiniLM-L6-v2")

    args = parser.parse_args()

    print(args)
    print("[Start] Making connection")

    main(args)

    print("[Done] Making connection")
    print(args)


''' Future work
Problem: Too slow to work this.

Solution:
It can be recoded efficiently.
1) gathering all keywords from csv files
2) mapping all keywords to wiki
3) connecting each csv file to wiki based on its own keyword

-> we can reduce redundant job.
'''