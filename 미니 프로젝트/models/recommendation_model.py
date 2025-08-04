import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

@st.cache_resource(show_spinner="🎬 TF-IDF 유사도 모델을 계산하는 중입니다...")
def get_tfidf_similarity_matrix(dataframe):
    tfidf = TfidfVectorizer(min_df=2)
    tfidf_matrix = tfidf.fit_transform(dataframe['text_for_tfidf'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

@st.cache_resource(show_spinner="✨ KoBERT 임베딩 및 유사도 모델을 계산하는 중입니다...")
def get_kobert_similarity_matrix(dataframe):
    model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    embeddings = model.encode(dataframe['text_for_kobert'].tolist(), convert_to_tensor=False, show_progress_bar=False) 
    return cosine_similarity(embeddings, embeddings)

def get_combined_recommendations(title, sim_matrix_tfidf, sim_matrix_kobert, title_to_index, top_n=5, weight_tfidf=0.5, weight_kobert=0.5):
    """
    TF-IDF와 KoBERT 유사도 행렬을 병합하여 영화를 추천하고, 추천 영화의 인덱스를 반환합니다.
    Args:
        title (str): 추천 기준이 될 영화 제목.
        sim_matrix_tfidf (np.array): TF-IDF 유사도 행렬.
        sim_matrix_kobert (np.array): KoBERT 유사도 행렬.
        title_to_index (pd.Series): 영화명에서 인덱스로 매핑하는 Series. (추가된 인자)
        top_n (int): 반환할 추천 영화의 수.
        weight_tfidf (float): TF-IDF 유사도에 대한 가중치 (0.0 ~ 1.0).
        weight_kobert (float): KoBERT 유사도에 대한 가중치 (0.0 ~ 1.0).
    Returns:
        list: 추천 영화의 인덱스 리스트.
    """
    idx = title_to_index.get(title)
    if idx is None: 
        st.warning(f"'{title}'에 대한 인덱스를 찾을 수 없습니다. 추천할 수 없습니다.")
        return None # 빈 리스트 대신 None을 반환하도록 하여 main_app.py에서 처리

    if idx >= len(sim_matrix_tfidf) or idx >= len(sim_matrix_kobert):
        st.error(f"'{title}'에 대한 인덱스({idx})가 유사도 모델 범위를 벗어납니다.")
        return None

    scores_tfidf = sim_matrix_tfidf[idx]
    scores_kobert = sim_matrix_kobert[idx]

    combined_scores = (scores_tfidf * weight_tfidf) + (scores_kobert * weight_kobert)

    sim_scores = sorted(list(enumerate(combined_scores)), key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    
    return movie_indices