import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import json
import requests
import torch
import time

response = requests.get('https://sbert.net/datasets/emnlp2016-2018.json')
papers = json.loads(response.text)

# https://www.sbert.net/docs/pretrained_models.html
model = SentenceTransformer('allenai-specter')

paper_texts = [paper['title'] + '[SEP]' + paper['abstract'] for paper in papers]

corpus_embeddings = torch.load("output.pt", map_location=torch.device('cpu'))


def search(tittle, abstractt):
    query_embedding = model.encode(tittle + '[SEP]' + abstractt, convert_to_tensor=True)

    search_hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=3)[0]

    for hit in search_hits:
        related_paper = papers[hit['corpus_id']]
        return st.write(related_paper)

# def stream_data(paper):
#     for word in paper:
#             yield word + " "
#             time.sleep(0.2)


st.image('images/Recomme!.png')
st.subheader("Recommendation System to provide similar research papers based on given title & abstract")

title = st.text_input('Enter title here')
abstract = st.text_input('Enter abstract here')

if st.button("Get Recommendations", type="primary"):
    with st.spinner('Wait for it...'):
        time.sleep(5)
    st.success('Done!')

    st.text("Recommendations shown based upon given title and abstract")
    search(title, abstract)
    st.balloons()
else:
    st.error("Please enter a valid title and abstract")
