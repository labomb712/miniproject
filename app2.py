import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import platform
from matplotlib import font_manager, rc
import requests # TMDB API 호출을 위해 추가

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# sentence_transformers 라이브러리가 설치되어 있어야 합니다.
# pip install -U sentence-transformers
from sentence_transformers import SentenceTransformer

# --- 1. 기본 설정 및 폰트 ---

def setup_korean_font():
    """
    운영체제에 맞는 한글 폰트를 설정합니다.
    """
    if platform.system() == 'Windows':
        font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
        rc('font', family=font_name)
    elif platform.system() == 'Darwin': # macOS
        rc('font', family='AppleGothic')
    else: # Linux
        font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        if font_manager.findfont(font_path):
            rc('font', family='NanumGothic')
        else:
            rc('font', family='DejaVu Sans')
    plt.rcParams['axes.unicode_minus'] = False

setup_korean_font()

st.set_page_config(page_title="영화 예측 시스템", layout="wide")
st.title("🎬 영화 예측 시스템")
st.markdown("---")

# --- 2. 데이터 및 API 관련 함수 ---

@st.cache_data(show_spinner="영화 데이터를 불러오는 중입니다...")
def load_data(file_path):
    """
    CSV 파일에서 영화 데이터를 로드하고 기본 전처리를 수행합니다.
    """
    df = pd.read_csv(file_path)
    for col in ['누적관객수', '누적매출액']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['개봉일'] = pd.to_datetime(df['개봉일'], errors='coerce', format='%Y-%m-%d')
    df.dropna(subset=['누적관객수', '누적매출액', '개봉일', '감독', '장르', '제작국가'], inplace=True)
    df['개봉_월'] = df['개봉일'].dt.month
    df['개봉_년'] = df['개봉일'].dt.year
    df['text_for_tfidf'] = df[['감독', '제작국가', '장르']].astype(str).agg(' '.join, axis=1)
    df['text_for_kobert'] = df.apply(
        lambda row: f"{row['감독']} 감독이 제작한 {row['제작국가']} 영화. 장르는 {row['장르']}이며, {row['개봉_년']}년 {row['개봉_월']}월에 개봉했습니다.",
        axis=1
    )
    return df

@st.cache_data(show_spinner=False) # API 호출 결과를 캐싱합니다.
def get_movie_poster_url(movie_title):
    """
    TMDB API를 사용하여 영화 포스터 URL을 가져옵니다.
    """
    API_KEY = "62fd419c4be9316756c61d72694907d3" # 제공된 API 키
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_title}&language=ko-KR"
    try:
        response = requests.get(search_url)
        response.raise_for_status() # 오류 발생 시 예외 처리
        data = response.json()
        if data['results']:
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except requests.exceptions.RequestException as e:
        # st.error(f"API 호출 중 오류 발생: {e}") # 디버깅 시 사용
        pass
    return "https://placehold.co/300x450/cccccc/000000?text=No+Image" # 이미지가 없을 경우

# 데이터 로드
df = load_data("data/20-25년_영화데이터_한글컬럼.csv")
title_to_index = pd.Series(df.index, index=df['영화명']).drop_duplicates()

# --- 3. 추천 모델 (TF-IDF & KoBERT) ---

@st.cache_resource(show_spinner="TF-IDF 유사도 모델을 계산하는 중입니다...")
def get_tfidf_similarity_matrix(dataframe):
    tfidf = TfidfVectorizer(min_df=2)
    tfidf_matrix = tfidf.fit_transform(dataframe['text_for_tfidf'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

@st.cache_resource(show_spinner="KoBERT 임베딩 및 유사도 모델을 계산하는 중입니다...")
def get_kobert_similarity_matrix(dataframe):
    model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    embeddings = model.encode(dataframe['text_for_kobert'].tolist(), convert_to_tensor=False, show_progress_bar=True)
    return cosine_similarity(embeddings, embeddings)

cosine_sim_tfidf = get_tfidf_similarity_matrix(df)
cosine_sim_kobert = get_kobert_similarity_matrix(df)

def get_recommendations(title, similarity_matrix, top_n=5):
    idx = title_to_index.get(title)
    if idx is None: return None
    sim_scores = sorted(list(enumerate(similarity_matrix[idx])), key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    
    # 추천된 영화 정보와 함께 포스터 URL도 가져옵니다.
    recommended_df = df.iloc[movie_indices][['영화명', '감독', '장르', '개봉일']].copy()
    recommended_df['포스터'] = recommended_df['영화명'].apply(get_movie_poster_url)
    return recommended_df[['포스터', '영화명', '감독', '장르', '개봉일']]

# --- 4. Streamlit UI - 영화 추천 ---

st.header("✨ 콘텐츠 기반 영화 추천")
st.write("영화를 선택하면 해당 영화의 포스터와 정보, 그리고 두 가지 방식으로 추천된 영화 목록을 보여줍니다.")

movie_list = ['영화를 선택하세요...'] + sorted(df['영화명'].unique().tolist())
selected_movie = st.selectbox("추천의 기준이 될 영화를 선택해주세요:", movie_list)

if selected_movie != '영화를 선택하세요...':
    st.markdown("---")
    movie_info = df[df['영화명'] == selected_movie].iloc[0]
    
    # 선택된 영화 정보 (포스터와 함께)
    st.subheader(f"'{selected_movie}' 정보")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(get_movie_poster_url(selected_movie), use_column_width=True)
    with col2:
        st.info(f"**감독:** {movie_info['감독']}")
        st.info(f"**장르:** {movie_info['장르']}")
        st.info(f"**제작국가:** {movie_info['제작국가']}")
        st.info(f"**개봉일:** {movie_info['개봉일'].date()}")
        st.info(f"**누적 관객수:** {int(movie_info['누적관객수']):,} 명")
        st.info(f"**누적 매출액:** ₩ {int(movie_info['누적매출액']):,}")

    st.markdown("---")
    st.subheader(f"'{selected_movie}'와 비슷한 영화 추천 목록")
    
    # 추천 결과 (포스터와 함께)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🤖 TF-IDF 기반 추천 (키워드 중심)")
        rec_tfidf = get_recommendations(selected_movie, cosine_sim_tfidf)
        if rec_tfidf is not None:
            st.data_editor(rec_tfidf, column_config={"포스터": st.column_config.ImageColumn("포스터")}, hide_index=True, use_container_width=True)
        else:
            st.warning("추천 결과를 찾을 수 없습니다.")
    with col2:
        st.markdown("#### 🧠 KoBERT 기반 추천 (의미 중심)")
        rec_kobert = get_recommendations(selected_movie, cosine_sim_kobert)
        if rec_kobert is not None:
            st.data_editor(rec_kobert, column_config={"포스터": st.column_config.ImageColumn("포스터")}, hide_index=True, use_container_width=True)
        else:
            st.warning("추천 결과를 찾을 수 없습니다.")

st.markdown("\n\n---\n\n")

# --- 5. Streamlit UI - 관객수 예측 ---

st.header("🎯 누적 관객수 예측 모델")
with st.spinner("관객수 예측 모델을 학습하는 중입니다..."):
    X = df[['누적매출액', '개봉_월', '장르', '제작국가']]
    y = df['누적관객수']
    numerical_features = ['누적매출액']
    categorical_features = ['개봉_월', '장르', '제작국가']
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)],
        remainder='passthrough')
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
    test_size_val = max(0.2, 1 / len(X) if len(X) > 0 else 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_val, random_state=42)
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    mse, rmse, r2 = mean_squared_error(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred)), r2_score(y_test, y_pred)

st.subheader("📊 모델 성능 지표")
col1, col2, col3 = st.columns(3)
col1.metric("MSE", f"{mse:,.0f}")
col2.metric("RMSE", f"{rmse:,.0f}")
col3.metric("R² Score", f"{r2:.4f}")

st.subheader("📈 실제 vs 예측 관객수 시각화")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, ax=ax, color='royalblue')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='이상적인 예측')
ax.set_xlabel("실제 누적 관객수")
ax.set_ylabel("예측 누적 관객수")
ax.set_title("랜덤 포레스트 회귀: 실제 vs 예측")
ax.legend()
ax.grid(True)
ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.xticks(rotation=45)
st.pyplot(fig)
