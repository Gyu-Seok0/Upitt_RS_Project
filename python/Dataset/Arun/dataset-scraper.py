import requests
from bs4 import BeautifulSoup
import pandas as pd
from os.path import exists
import json

from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

model_name ='all-MiniLM-L6-v2' 
model1 = SentenceTransformer(model_name)
kw_model = KeyBERT(model=model1)

texts = []
file_map = {}

def scrape_page(url):

    try:
        response = requests.get(url, timeout=1)

        if response.status_code == 200:

            soup = BeautifulSoup(response.content,'html.parser')

            page_text = soup.text.strip().replace('\n','').replace('\t','')

            keywords = kw_model.extract_keywords(page_text, stop_words = 'english', use_maxsum = True, top_n = 15)

            return soup, [k[0] for k in keywords]
    
    except Exception as e:
        print(e)

def scrape_main_links(links,starting_url):
    for link in links:
        url = link.get('href')

        if url.split('.')[1] == 'html':
            soup, text_ks = scrape_page(f"{starting_url}/{url}")

            texts.append((links.index(link), text_ks))

        file_map[links.index(link)] = soup.title.text if not soup.title == "None" else ' '.join(soup.text.split()[:5])

def get_main_links(url):

    response = requests.get(url,timeout=1)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.find_all('a')



def main():
    starting_url = "https://nlp.stanford.edu/IR-book/html/htmledition"

    links = get_main_links(f"{starting_url}/contents-1.html")

    scrape_main_links(links[12:-14],starting_url)
    
    json.dump(file_map, open('file_map.json','w+'))

    pd.DataFrame(texts, columns = ['csv_file','keywords']).to_csv('./test.csv',index=False)


if __name__ == "__main__":
    main()


