from tempfile import NamedTemporaryFile
import pandas as pd
import streamlit as st
import numpy as np
# from app import *
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
nltk.download('stopwords')
stop_words = stopwords.words('english')
# from constant import model
from llama_index.core import SimpleDirectoryReader
import os
from scipy.spatial.distance import cosine
import fitz
from docx import Document
import glob
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained("bert-base-uncased")
# from flask import Flask
# Load the SentenceTransformer model
from sentence_transformers import SentenceTransformer
model1 = SentenceTransformer("aspire/acge_text_embedding")
# sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
# model1=SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')

# app = Flask(__name__)

# @app.route('/')
def preprocessing(document):
    """Preprocesses text data."""
    text1 = document[0].text.replace('\n', '').lower()
    text = re.sub(r'[^\x00-\x7F]+', ' ', text1)  # Remove non-ASCII characters
    tokens = word_tokenize(text)
    tokens = [re.sub(r'[^a-zA-Z\s]', '', token) for token in tokens]  # Remove punctuation
    tokens = [token for token in tokens if token]  # Remove empty tokens
    filtered_tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
    preprocessed_text = ' '.join(filtered_tokens)
    return preprocessed_text

# def cosine(u, v):
#     return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

# def embedding_model(data):
#     """Encodes data using the provided model.

#     Args:
#         data: The text data to encode.

#     Returns:
#         The encoded representation of the data.
#     """
#     encoded_input = tokenizer(data, return_tensors='pt',max_length=512,truncation=True)
#     with torch.no_grad():
#         output = model(**encoded_input)
    # output=model.encode(data, normalize_embeddings=True)
    # return output

def embedding_model1(data):
    """Encodes data using the provided model.

    Args:
        data: The text data to encode.

    Returns:
        The encoded representation of the data.
    """
    # encoded_input = tokenizer(data, return_tensors='pt',max_length=512,truncation=True)
    # with torch.no_grad():
    #     output = model(**encoded_input)
    # 
    output=model1.encode(data)
    print('out',output.shape)
    return output


def jd_embedding(uploaded_jd_file):
    """Computes the embedding for a job description (JD) file."""
    with NamedTemporaryFile(delete=False) as tmp_file:
        tmp_filename = tmp_file.name
        tmp_file.write(uploaded_jd_file.read())
    document = SimpleDirectoryReader(input_files=[tmp_filename]).load_data()
    data = preprocessing(document)
    # JD_embedding = embedding_model(data)
    print('data',data)
    JD_embedding1 = embedding_model1(data)
    print('jd',JD_embedding1.shape)
    os.unlink(tmp_filename)  # Delete temporary file
    return JD_embedding1

def debug_app():
    st.markdown('''# TalentMatch360 🌟''')
                
    st.markdown('''### Nextgen tool for evaluating Job Descriptions and Resumes''')

    left_column, center_column, right_column = st.columns(3)

    with left_column:
        uploaded_jd_file = st.file_uploader("Upload your JD here")
        if uploaded_jd_file is not None:
            try:
                # JD_embedding = jd_embedding(uploaded_jd_file)[0]
                JD_embedding1 = jd_embedding(uploaded_jd_file)
                # print('JD 000',JD_embedding)
                st.write("JD uploaded")
            except Exception as e:
                st.error(f"Error processing JD: {e}")
    
    with right_column:
        uploaded_resume_files = st.file_uploader(
            "Upload all of the resumes", accept_multiple_files=True)
        resume_embeddings = {}  # Dictionary to store resume embeddings
        resume_embeddings1={}

        if uploaded_resume_files:
            resume_embeddings.clear()
            for uploaded_resume_file in uploaded_resume_files:
                if uploaded_resume_file is not None:
                    try:
                        with NamedTemporaryFile(delete=False) as tmp_file:
                            tmp_filename = tmp_file.name
                            tmp_file.write(uploaded_resume_file.read())
                        document = SimpleDirectoryReader(input_files=[tmp_filename]).load_data()
                        data = preprocessing(document)
                        # REsume_embedding = embedding_model(data)
                        print('data',data)
                        REsume_embedding1 = embedding_model1(data)
                        print('resume',REsume_embedding1.shape)
                        # resume_embeddings[uploaded_resume_file.name] = REsume_embedding
                        resume_embeddings1[uploaded_resume_file.name] = REsume_embedding1
                        os.unlink(tmp_filename)  # Delete temporary file
                    except Exception as e:
                        st.error(f"Error processing resume {uploaded_resume_file.name}: {e}")
            st.write("Resume uploaded")

    with center_column:
        score_dict = {}
        # score_dict1={}
        for filename, resume_embedding in resume_embeddings1.items():
            # cosine_JD=cosine_similarity(JD_embedding['pooler_output'], resume_embedding['pooler_output'])
            # similarity1 = JD_embedding1 @ resume_embedding.T
            # JD_embedding1 = JD_embedding1.reshape(1,1)
            # resume_embedding=resume_embedding.reshape(384,1)
            score = 1 - cosine(JD_embedding1, resume_embedding)
            # print('cosine1',similarity1)
            similarity_score_percentage = score* 100
            score_dict[filename] = similarity_score_percentage                
        
        # for filename, resume_embedding in resume_embeddings.items():
        #     similarity2 = cosine(JD_embedding, resume_embedding)
        #     similarity_score_percentage1 = similarity2* 100
        #     score_dict1[filename] = similarity_score_percentage1
            
        sorted_dict_desc = dict(sorted(score_dict.items(), key=lambda item: item[1], reverse=True))
        df = pd.DataFrame(list(sorted_dict_desc.items()), columns=['Resume', 'Score'])

        # sorted_dict_desc1 = dict(sorted(score_dict1.items(), key=lambda item: item[1], reverse=True))
        # df1 = pd.DataFrame(list(sorted_dict_desc1.items()), columns=['Resume', 'Score'])
        
        # st.dataframe(df, use_container_width=True, width=1200,hide_index=True)
        st.dataframe(df,hide_index=True)

if __name__ == "__main__":
    # app.run(host='0.0.0.0', port=3001, debug=True)
    debug_app()
