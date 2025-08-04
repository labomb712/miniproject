import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import platform
from matplotlib import font_manager, rc
import requests
import datetime
import os

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
        if os.path.exists(font_path):
            rc('font', family='NanumGothic')
        else:
            st.warning("나눔고딕 폰트가 시스템에 없습니다. 다른 폰트로 대체됩니다. 폰트 설치: sudo apt-get install fonts-nanum*")
            rc('font', family='DejaVu Sans')
    plt.rcParams['axes.unicode_minus'] = False

setup_korean_font()

st.set_page_config(page_title="영화 예측 시스템", layout="centered", initial_sidebar_state="expanded") # 사이드바 기본 확장

# --- 사용자 정의 CSS (새로운 추천 목록 디자인 적용 및 라디오 버튼 간격 추가) ---
st.markdown("""
<style>
    /* 전체 앱 배경 및 기본 텍스트 색상 */
    .stApp {
        background-color: #2c313d; /* Soft dark gray */
        color: #f0f0f0; /* Very light gray for main text */
    }

    /* 사이드바 배경 */
    .stSidebar {
        background-color: #20232a; /* Slightly darker gray for sidebar */
    }

    /* 사이드바 헤더 및 라벨 텍스트 색상 (메인 타이틀과 통일) */
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar h5, .stSidebar h6,
    .stSidebar .stSelectbox label, .stSidebar .stDateInput label, .stSidebar .stRadio label,
    .stSidebar .stSlider label, .stSidebar .stTextInput label { /* 추가적으로 다른 위젯 라벨도 포함 */
        color: #FFD700; /* Gold/Yellow for consistency */
    }
    /* 사이드바 일반 텍스트 */
    .stSidebar .stMarkdown p, .stSidebar .stMarkdown strong, .stSidebar label {
        color: #e0e0e0; /* Light gray for general text in sidebar */
    }

    /* 사이드바 라디오 버튼 텍스트 색상 (선택 안 된 상태) */
    .stSidebar .stRadio div[role="radiogroup"] label p {
        color: #f0f0f0; /* 기본 라디오 버튼 텍스트 색상 */
    }
    /* 사이드바 라디오 버튼 텍스트 색상 (선택된 상태) */
    .stSidebar .stRadio div[role="radiogroup"] label[data-baseweb="radio"] span:first-child p {
        color: #2c313d !important; /* 선택되었을 때 배경색에 대비되는 어두운 색 */
        font-weight: bold; /* 선택된 항목 더 강조 */
    }
    /* 사이드바 라디오 버튼 호버 시 텍스트 색상 */
    .stSidebar .stRadio div[role="radiogroup"] label:hover p {
        color: #2c313d !important; /* 호버 시 배경색에 대비되는 어두운 색 */
    }
    /* 라디오 버튼 항목 사이 간격 추가 (horizontal=True일 때 유효) */
    .stSidebar .stRadio div[role="radiogroup"] {
        display: flex; /* 자식 요소들을 가로로 정렬 */
        justify-content: flex-start; /* flex-start로 변경하여 왼쪽부터 고정 간격 */
        gap: 20px; /* 각 항목 사이의 간격 */
        flex-wrap: wrap; /* 공간이 부족하면 다음 줄로 넘어가도록 */
    }
    .stSidebar .stRadio div[role="radiogroup"] label {
        margin-right: 0; /* 기존 margin-right 충돌 방지 */
    }


    /* 메인 컨텐츠 헤더 및 타이틀 색상 */
    h1, h2, h3, h4, h5, h6 {
        color: #FFD700; /* Gold/Yellow for accents */
    }
    
    /* 메인 타이틀 설명 텍스트 */
    .stMarkdown p {
        color: #e0e0e0; /* Slightly darker than main text for general info */
    }

    /* selectbox, button 등 위젯 배경 */
    .stSelectbox > div > div, .stTextInput > div > div > input, .stDateInput > div > div > input, .stRadio > label, .stButton > button {
        background-color: #3f4451; /* Medium dark gray for widgets */
        color: #f0f0f0;
        border: 1px solid #FFD700; /* Gold border */
        border-radius: 5px;
    }
    
    /* Selectbox 드롭다운 아이템 */
    .stSelectbox div[role="listbox"] {
        background-color: #3f4451;
        color: #f0f0f0;
    }
    .stSelectbox div[role="option"] {
        color: #f0f0f0;
    }
    .stSelectbox div[role="option"]:hover {
        background-color: #FFD700;
        color: #2c313d; /* Dark text on hover */
    }


    .stButton > button:hover {
        background-color: #FFD700; /* Gold on hover */
        color: #2c313d; /* Dark text on hover */
        border: 1px solid #f0f0f0; /* Light border on hover */
    }

    /* Metric 카드 */
    [data-testid="stMetric"] {
        background-color: #3f4451; /* Darker background for metrics */
        border: 1px solid #FFD700; /* Gold border */
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Subtle shadow */
    }
    [data-testid="stMetricLabel"] {
        color: #e0e0e0; /* Light gray label */
    }
    [data-testid="stMetricValue"] {
        color: #f0f0f0; /* White value */
        font-size: 2em; /* Make value larger */
        font-weight: bold;
    }
    [data-testid="stMetricDelta"] {
        color: #FFD700; /* Delta in accent color */
    }


    /* 경고/에러 메시지 */
    .stAlert {
        background-color: #5c2020; /* Darker red for errors */
        color: #ffebeb;
        border-left: 5px solid #ff4d4d;
        border-radius: 5px;
    }
    .stWarning {
        background-color: #5c4520; /* Darker orange for warnings */
        color: #fff8eb;
        border-left: 5px solid #ffcc66;
        border-radius: 5px;
    }
    
    /* 기존 recommendation-card 스타일 제거 또는 비활성화 */
    /* div[data-testid^="column"] > div {
        display: none; // 기존 컬럼 기반 카드는 숨김
    } */

    /* 새로운 추천 목록 스타일 (각 영화가 하나의 리스트 아이템처럼 보이도록) */
    .recommendation-list-item {
        display: flex; /* 가로 배열을 위해 flexbox 사용 */
        align-items: flex-start; /* 상단 정렬 */
        background-color: #3f4451; /* 리스트 아이템 배경색 */
        border-radius: 8px; /* 모서리 둥글게 */
        padding: 15px; /* 내부 여백 */
        margin-bottom: 12px; /* 각 아이템 간 간격 */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* 은은한 그림자 */
        border-left: 5px solid #FFD700; /* 강조를 위한 좌측 골드 바 */
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out; /* 부드러운 호버 효과 */
    }

    .recommendation-list-item:hover {
        transform: translateY(-3px); /* 호버 시 약간 위로 */
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4); /* 그림자 진하게 */
    }

    .recommendation-list-item img {
        width: 80px; /* 포스터 너비 */
        height: 120px; /* 포스터 높이 (비율 유지) */
        object-fit: cover;
        border-radius: 5px;
        margin-right: 15px; /* 포스터와 텍스트 사이 간격 */
        flex-shrink: 0; /* 이미지 크기 고정 */
        border: 1px solid #5a5f6e;
    }

    .movie-info-container {
        display: flex;
        flex-direction: column;
        text-align: left; /* 텍스트 좌측 정렬 */
        flex-grow: 1; /* 남은 공간을 채우도록 */
    }

    .movie-title-list {
        font-size: 18px; /* 제목 크기 키움 */
        color: #FFD700; /* 제목 색상 */
        font-weight: bold;
        margin-bottom: 5px;
        line-height: 1.3;
    }

    .movie-detail-list {
        font-size: 14px; /* 상세 정보 텍스트 크기 */
        color: #e0e0e0; /* 상세 정보 텍스트 색상 */
        margin-bottom: 3px;
        line-height: 1.3;
    }
    .movie-detail-list strong {
        color: #f0f0f0; /* 레이블 강조 */
    }
    /* 마지막 항목의 하단 마진 제거 */
    .movie-info-container .movie-detail-list:last-of-type {
        margin-bottom: 0;
    }


    /* Matplotlib plot 배경을 Streamlit 앱 배경과 동일하게 설정 */
    .stPlotlyChart {
        background-color: #2c313d; /* Match app background */
        border-radius: 10px;
        padding: 10px;
    }

    /* 영화 상세 정보 창 조절 */
    .movie-detail-box {
        background-color: #3f4451;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        height: 100%; /* Ensure it fills column height if content is short */
        display: flex;
        flex-direction: column;
        justify-content: center; /* Vertically center content if space allows */
    }
    .movie-detail-item {
        margin-bottom: 8px; /* Spacing between detail items */
        font-size: 28px !important; /* Increased size and added !important */
        font-weight: bold !important; /* Made bolder and added !important */
        line-height: 1.4; /* 줄 간격 조절 */
    }
</style>
""", unsafe_allow_html=True)


st.title("🎬 영화 예측 시스템: 당신의 다음 영화를 발견하세요!")
st.markdown("""
    <p style="font-size:20px; text-align: center;">
    "영화는 우리에게 꿈을 꾸게 합니다." 🌟 
    <br>원하는 영화를 탐색하고, 새로운 추천을 받으며, 흥행 성과를 예측해보세요.
    </p>
    """, unsafe_allow_html=True)
st.markdown("---")

# --- 2. 데이터 및 API 관련 함수 ---

@st.cache_data(show_spinner="🎞️ 영화 데이터를 불러오는 중입니다...")
def load_data(file_path):
    """
    CSV 파일에서 영화 데이터를 로드하고 기본 전처리를 수행합니다.
    """
    if not os.path.exists(file_path):
        st.error(f"오류: 데이터 파일 '{file_path}'을(를) 찾을 수 없습니다. 'data' 폴더에 파일을 넣어주세요.")
        st.stop()

    df = pd.read_csv(file_path)
    
    for col in ['누적관객수', '누적매출액']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['개봉일'] = pd.to_datetime(df['개봉일'], errors='coerce', format='%Y-%m-%d') 
    
    df['감독'].fillna('알 수 없음', inplace=True) 
    df['제작국가'].fillna('알 수 없음', inplace=True)
    df['장르'].fillna('알 수 없음', inplace=True)

    df['누적관객수'].fillna(0, inplace=True)
    df['누적매출액'].fillna(0, inplace=True)

    df.dropna(subset=['개봉일', '영화명'], inplace=True) 
    df.reset_index(drop=True, inplace=True)
    
    df['개봉년도'] = df['개봉일'].dt.year
    df['개봉월'] = df['개봉일'].dt.month
    df['개봉요일'] = df['개봉일'].dt.weekday 

    df['개봉년도'] = df['개봉년도'].fillna(0).astype(int)
    df['개봉월'] = df['개봉월'].fillna(0).astype(int)
    df['개봉요일'] = df['개봉요일'].fillna(0).astype(int)
    
    df['text_for_tfidf'] = df[['감독', '제작국가', '장르']].astype(str).agg(' '.join, axis=1)
    df['text_for_kobert'] = df.apply(
        lambda row: f"{row['감독']} 감독이 제작한 {row['제작국가']} 영화. 장르는 {row['장르']}이며, {row['개봉년도']}년 {row['개봉월']}월에 개봉했습니다.",
        axis=1
    )
    return df

@st.cache_data(show_spinner=False)
def get_movie_poster_url(movie_title):
    """
    TMDB API를 사용하여 영화 포스터 URL을 가져옵니다.
    """
    API_KEY = "62fd419c4be9316756c61d72694907d3" 
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_title}&language=ko-KR"
    try:
        response = requests.get(search_url)
        response.raise_for_status()
        data = response.json()
        if data['results']:
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except requests.exceptions.RequestException as e:
        pass
    return "https://placehold.co/300x450/cccccc/000000?text=No+Image"

# 데이터 로드
DATA_FILE_PATH = "data/청불제거_최종_DB컬럼.csv"
df = load_data(DATA_FILE_PATH)

# 데이터가 비어있을 경우 Early Exit
if df.empty:
    st.error("데이터 로드 및 전처리 후 데이터가 비어 있습니다. 파일 내용과 전처리 조건을 확인해주세요.")
    st.stop()

title_to_index = pd.Series(df.index, index=df['영화명']).drop_duplicates() 

# --- 3. 추천 모델 (TF-IDF & KoBERT) ---

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

# 유사도 행렬 계산
cosine_sim_tfidf = get_tfidf_similarity_matrix(df)
cosine_sim_kobert = get_kobert_similarity_matrix(df)

def get_combined_recommendations(title, sim_matrix_tfidf, sim_matrix_kobert, top_n=5, weight_tfidf=0.5, weight_kobert=0.5):
    """
    TF-IDF와 KoBERT 유사도 행렬을 병합하여 영화를 추천합니다.
    """
    idx = title_to_index.get(title)
    if idx is None: 
        st.warning(f"'{title}'에 대한 인덱스를 찾을 수 없습니다. 추천할 수 없습니다.")
        return None
    
    if idx >= len(sim_matrix_tfidf) or idx >= len(sim_matrix_kobert):
        st.error(f"'{title}'에 대한 인덱스({idx})가 유사도 모델 범위를 벗어납니다.")
        return None

    scores_tfidf = sim_matrix_tfidf[idx]
    scores_kobert = sim_matrix_kobert[idx]

    combined_scores = (scores_tfidf * weight_tfidf) + (scores_kobert * weight_kobert)

    sim_scores = sorted(list(enumerate(combined_scores)), key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    
    recommended_df = df.iloc[movie_indices][['영화명', '감독', '장르', '개봉일']].copy()
    recommended_df['포스터'] = recommended_df['영화명'].apply(get_movie_poster_url)
    return recommended_df[['포스터', '영화명', '감독', '장르', '개봉일']]


# --- 사이드바 추가 ---
st.sidebar.header("🔍 영화 검색 및 필터")
st.sidebar.markdown("---")

# 감독 필터
all_directors = ['전체 감독'] + sorted(df['감독'].unique().tolist())
selected_director = st.sidebar.selectbox("감독:", all_directors)

# 장르 필터
all_genres = ['전체 장르'] + sorted(df['장르'].unique().tolist())
selected_genre = st.sidebar.selectbox("장르:", all_genres)

# 개봉일 범위 검색
st.sidebar.markdown("---")
st.sidebar.subheader("📅 개봉일 범위")

min_date_for_display = df['개봉일'].min().date() if not df.empty else datetime.date(2000, 1, 1)
max_date_for_display = df['개봉일'].max().date() if not df.empty else datetime.date.today()

start_date = st.sidebar.date_input(
    "시작일:", 
    value=min_date_for_display, 
    min_value=min_date_for_display, 
    max_value=max_date_for_display, 
    key="sidebar_start_date"
)
end_date = st.sidebar.date_input(
    "종료일:", 
    value=max_date_for_display, 
    min_value=min_date_for_display, 
    max_value=max_date_for_display, 
    key="sidebar_end_date"
)

date_filter_valid = True
if start_date > end_date:
    st.sidebar.error("⚠️ 시작 개봉일은 종료 개봉일보다 빠를 수 없습니다.")
    date_filter_valid = False


# 필터링된 영화 목록 생성
filtered_df = df.copy()

if selected_director != '전체 감독':
    filtered_df = filtered_df[filtered_df['감독'] == selected_director] 

if selected_genre != '전체 장르':
    filtered_df = filtered_df[filtered_df['장르'] == selected_genre]

if date_filter_valid:
    filtered_df = filtered_df[
        (filtered_df['개봉일'].dt.date >= start_date) & 
        (filtered_df['개봉일'].dt.date <= end_date)
    ]

if filtered_df.empty and (selected_director != '전체 감독' or selected_genre != '전체 장르' or not date_filter_valid):
    st.warning("선택하신 조건에 해당하는 영화를 찾을 수 없습니다. 필터를 초기화하거나 다른 조건을 시도해보세요.")
    movie_list = ['영화를 선택하세요...']
    selected_movie = '영화를 선택하세요...'
elif not filtered_df.empty:
    movie_list = ['영화를 선택하세요...'] + sorted(filtered_df['영화명'].unique().tolist()) 
    if 'selected_movie' not in st.session_state or st.session_state.selected_movie not in movie_list:
        selected_movie = '영화를 선택하세요...'
    else:
        selected_movie = st.session_state.selected_movie
else: 
    movie_list = ['영화를 선택하세요...'] + sorted(df['영화명'].unique().tolist()) 
    if 'selected_movie' not in st.session_state:
        selected_movie = '영화를 선택하세요...'
    else:
        selected_movie = st.session_state.selected_movie


st.markdown("<strong><p style='font-size:22px;'>🔍 추천의 기준이 될 영화를 선택해주세요:</p></strong>", unsafe_allow_html=True)
selected_movie = st.selectbox("", movie_list, key="main_movie_selector", label_visibility="collapsed") 
st.session_state.selected_movie = selected_movie

# --- 4. Streamlit UI - 영화 추천 ---

st.header("✨ 콘텐츠 기반 영화 추천")
st.write("아래에서 영화를 선택하면 해당 영화의 상세 정보와 함께 비슷한 영화들을 추천해 드립니다.")

if selected_movie != '영화를 선택하세요...':
    st.markdown("---") 
    movie_info_rows = df[df['영화명'] == selected_movie] 
    
    if not movie_info_rows.empty:
        movie_info = movie_info_rows.iloc[0]
        
        st.subheader(f"[{selected_movie}] 상세 정보")
        
        # 컬럼 비율 조정: 포스터(1), 정보(2)
        col1, col2 = st.columns([1, 2]) 
        
        with col1:
            st.image(get_movie_poster_url(selected_movie), width=200, caption=f"'{selected_movie}' 포스터") 
        with col2:
            st.markdown(
                f"""
                <div class="movie-detail-box">
                    <p class="movie-detail-item"><strong>감독:</strong> {movie_info['감독']}</p>
                    <p class="movie-detail-item"><strong>장르:</strong> {movie_info['장르']}</p>
                    <p class="movie-detail-item"><strong>제작국가:</strong> {movie_info['제작국가']}</p>
                    <p class="movie-detail-item"><strong>개봉일:</strong> {movie_info['개봉일'].date()}</p>
                    <p class="movie-detail-item"><strong>누적 관객수:</strong> {int(movie_info['누적관객수']):,} 명</p>
                    <p class="movie-detail-item"><strong>누적 매출액:</strong> ₩ {int(movie_info['누적매출액']):,}</p>
                </div>
                """, unsafe_allow_html=True
            )

        st.markdown("---")
        st.subheader(f"[{selected_movie}]와 비슷한 영화 추천 목록 🍿")
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚖️ 추천 기준 조정")
        
        recommendation_mode = st.sidebar.radio(
            "어떤 기준으로 추천하시겠어요?",
            ('의미 중심 (KoBERT)', '키워드 중심 (TF-IDF)'), # '중간 (50:50)' 제거
            index=0, # 기본 선택을 '의미 중심 (KoBERT)'으로 변경 (인덱스 0)
            key="recommendation_mode",
            horizontal=True
        )

        weight_tfidf = 0.5 
        if recommendation_mode == '의미 중심 (KoBERT)':
            weight_tfidf = 0.0 # KoBERT 중심이므로 TF-IDF 가중치 0
        elif recommendation_mode == '키워드 중심 (TF-IDF)':
            weight_tfidf = 1.0 # TF-IDF 중심이므로 TF-IDF 가중치 1
        
        weight_kobert = 1.0 - weight_tfidf 
        
        st.markdown("<p style='font-size:20px;'><b>👍 당신을 위한 추천 영화들:</b></p>", unsafe_allow_html=True)
        rec_combined = get_combined_recommendations(
            selected_movie, 
            cosine_sim_tfidf, 
            cosine_sim_kobert, 
            top_n=5, 
            weight_tfidf=weight_tfidf, 
            weight_kobert=weight_kobert
        )
        if rec_combined is not None and not rec_combined.empty:
            # 새로운 목록형 디자인 적용
            for i, row in rec_combined.iterrows():
                st.markdown(
                    f"""
                    <div class="recommendation-list-item">
                        <img src="{row['포스터']}" alt="{row['영화명']} 포스터">
                        <div class="movie-info-container">
                            <div class="movie-title-list">{row['영화명']}</div>
                            <div class="movie-detail-list"><strong>감독:</strong> {row['감독']}</div>
                            <div class="movie-detail-list"><strong>장르:</strong> {row['장르']}</div>
                            <div class="movie-detail-list"><strong>개봉일:</strong> {row['개봉일'].date()}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True
                )
        else:
            st.warning("융합 추천 결과를 찾을 수 없습니다.")
    else:
        st.error(f"선택한 영화 '{selected_movie}'의 정보를 데이터에서 찾을 수 없습니다.")


st.markdown("\n\n---\n\n")

# --- 5. Streamlit UI - 누적 관객수 예측 (XGBoost 모델) ---

st.header("📈 누적 관객수 예측 모델 (XGBoost)")
st.write("이 섹션에서는 XGBoost 모델을 사용하여 영화의 누적 관객수를 예측하고, 모델의 성능을 시각화합니다.")

with st.spinner("⏳ 관객수 예측 모델을 학습하는 중입니다..."):
    xgb_features = ['감독', '제작국가', '장르', '개봉년도', '개봉월', '개봉요일', '누적매출액']
    xgb_target = '누적관객수'

    required_for_xgb = xgb_features + [xgb_target]
    if not all(col in df.columns for col in required_for_xgb):
        missing_cols = [col for col in required_for_xgb if col not in df.columns]
        st.error(f"XGBoost 모델 학습에 필요한 다음 컬럼이 없습니다: {', '.join(missing_cols)}. 데이터 파일을 확인해주세요.")
        fig, ax = plt.subplots(figsize=(10, 6))
        # Ensure plot background matches app background for seamless integration
        fig.patch.set_facecolor('#2c313d') 
        ax.set_facecolor('#2c313d')
        ax.text(0.5, 0.5, "필수 데이터 컬럼 누락", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16, color='#FFD700')
        ax.axis('off')
        st.pyplot(fig)
        st.stop() 

    xgb_df = df.copy()

    le = LabelEncoder() 
    
    for col in ['감독', '제작국가', '장르']:
        xgb_df[col] = le.fit_transform(xgb_df[col].astype(str)) 

    X_xgb = xgb_df[xgb_features]
    y_xgb = xgb_df[xgb_target]

    if X_xgb.empty or len(X_xgb) < 2:
        st.warning("XGBoost 모델 학습을 위한 데이터가 충분하지 않습니다. 파일 내용과 전처리 결과를 확인해주세요.")
        mse, rmse, r2 = 0, 0, 0
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#2c313d') 
        ax.set_facecolor('#2c313d')
        ax.text(0.5, 0.5, "데이터 부족으로 예측 불가", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16, color='#FFD700')
        ax.axis('off')
    else:
        try:
            from sklearn.model_selection import train_test_split as train_train_split # Changed from train_test_split to avoid name collision if not imported directly
            X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_train_split(
                X_xgb, y_xgb, test_size=0.2, random_state=42
            )

            dtrain_xgb = xgb.DMatrix(X_train_xgb, label=y_train_xgb)
            dtest_xgb = xgb.DMatrix(X_test_xgb, label=y_test_xgb)

            params_xgb = {
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.9,
                'colsample_bytree': 0.7,
                'reg_alpha': 0.5,
                'reg_lambda': 1.0,
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'seed': 42
            }

            model_xgb = xgb.train(
                params_xgb,
                dtrain_xgb,
                num_boost_round=1000,
                evals=[(dtrain_xgb, 'train'), (dtest_xgb, 'valid')],
                early_stopping_rounds=50,
                verbose_eval=False 
            )

            y_pred_xgb = model_xgb.predict(dtest_xgb)
            y_pred_xgb[y_pred_xgb < 0] = 0 

            mse = mean_squared_error(y_test_xgb, y_pred_xgb)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_xgb, y_pred_xgb)
            mae = mean_absolute_error(y_test_xgb, y_pred_xgb) 

            st.subheader("📊 모델 성능 지표")
            col1, col2, col3, col4 = st.columns(4) 
            col1.metric("MSE", f"{mse:,.0f}")
            col2.metric("RMSE", f"{rmse:,.0f}")
            col3.metric("MAE", f"{mae:,.0f}") 
            col4.metric("R² Score", f"{r2:.4f}")

            st.subheader("📉 실제 vs 예측 관객수 시각화 (XGBoost 모델)")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=y_test_xgb, y=y_pred_xgb, alpha=0.6, ax=ax, color='#66b2ff') # Lighter blue scatter
            ax.plot([y_test_xgb.min(), y_test_xgb.max()], [y_test_xgb.min(), y_test_xgb.max()], 'r--', lw=2, label='이상적인 예측')
            ax.set_xlabel("실제 누적 관객수", color='#f0f0f0')
            ax.set_ylabel("예측 누적 관객수", color='#f0f0f0')
            ax.set_title("XGBoost 회귀: 실제 vs 예측", color='#FFD700')
            ax.legend(labelcolor='#f0f0f0')
            ax.grid(True, color='#5a5f6e', linestyle=':', alpha=0.7) # Lighter grid lines
            
            # Set tick and spine colors for the plot
            ax.tick_params(axis='x', colors='#f0f0f0')
            ax.tick_params(axis='y', colors='#f0f0f0')
            ax.spines['left'].set_color('#f0f0f0')
            ax.spines['bottom'].set_color('#f0f0f0')
            ax.spines['right'].set_color('#f0f0f0')
            ax.spines['top'].set_color('#f0f0f0')
            
            # Set plot background to match app background
            fig.patch.set_facecolor('#2c313d')
            ax.set_facecolor('#2c313d')

            ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
            ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
            plt.xticks(rotation=45)
        except Exception as e: 
            st.error(f"XGBoost 모델 학습 또는 예측 중 오류 발생: {e}. 데이터셋 크기 또는 특성을 확인해주세요.")
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#2c313d') 
            ax.set_facecolor('#2c313d')
            ax.text(0.5, 0.5, "모델 학습 중 오류 발생", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16, color='#FFD700')
            ax.axis('off')
    st.pyplot(fig)

# --- 6. 'merged_test.csv' 파일의 예측 결과 시각화 추가 ---
st.markdown("\n\n---\n\n")
st.header("📊 CatBoost 예측 결과 시각화") # CatBoost로 명칭 변경 유지
st.write("별도로 예측된 'merged_test.csv' 파일의 CatBoost 모델 예측 결과를 시각화합니다.")

MERGED_TEST_FILE_PATH = "data/merged_test.csv"

@st.cache_data(show_spinner="⏳ CatBoost 예측 결과 데이터를 불러오는 중입니다...")
def load_and_preprocess_merged_test_data(file_path):
    if not os.path.exists(file_path):
        st.error(f"오류: 데이터 파일 '{file_path}'을(를) 찾을 수 없습니다.")
        return pd.DataFrame() 

    merged_df = pd.read_csv(file_path)
    
    for col in ['누적관객수', '예측_누적관객수']:
        if col not in merged_df.columns:
            st.error(f"'{col}' 컬럼이 '{file_path}' 파일에 없습니다.")
            return pd.DataFrame()
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').fillna(0)
    
    return merged_df

merged_test_df = load_and_preprocess_merged_test_data(MERGED_TEST_FILE_PATH)

# --- 디버깅용 코드 시작 (이전 요청에서 추가된 부분, 필요 없으면 삭제 가능) ---
# st.write(f"merged_test_df가 비어있습니까?: {merged_test_df.empty}")
# if not merged_test_df.empty:
#     st.write("merged_test_df의 첫 5행:")
#     st.dataframe(merged_test_df.head())
#     st.write("merged_test_df 컬럼:")
#     st.write(merged_test_df.columns.tolist())
#     st.write("누적관객수 데이터 타입:", merged_test_df['누적관객수'].dtype)
#     st.write("예측_누적관객수 데이터 타입:", merged_test_df['예측_누적관객수'].dtype)
# --- 디버깅용 코드 끝 ---

if not merged_test_df.empty:
    y_actual_merged = merged_test_df['누적관객수']
    y_predicted_merged = merged_test_df['예측_누적관객수']

    y_predicted_merged[y_predicted_merged < 0] = 0

    st.subheader("📉 실제 누적관객수 vs 예측 누적관객수 (CatBoost 모델)")
    fig_merged, ax_merged = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=y_actual_merged, y=y_predicted_merged, alpha=0.6, ax=ax_merged, color='#85e085') # Lighter green scatter
    
    min_val = min(y_actual_merged.min(), y_predicted_merged.min())
    max_val = max(y_actual_merged.max(), y_predicted_merged.max())
    
    ax_merged.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='이상적인 예측')
    ax_merged.set_xlabel("실제 누적 관객수", color='#f0f0f0')
    ax_merged.set_ylabel("예측 누적 관객수", color='#f0f0f0')
    ax_merged.set_title("CatBoost 회귀: 실제 vs 예측", color='#FFD700')
    ax_merged.legend(labelcolor='#f0f0f0')
    ax_merged.grid(True, color='#5a5f6e', linestyle=':', alpha=0.7) # Lighter grid lines
    
    # Set tick and spine colors for the plot
    ax_merged.tick_params(axis='x', colors='#f0f0f0')
    ax_merged.tick_params(axis='y', colors='#f0f0f0')
    ax_merged.spines['left'].set_color('#f0f0f0')
    ax_merged.spines['bottom'].set_color('#f0f0f0')
    ax_merged.spines['right'].set_color('#f0f0f0')
    ax_merged.spines['top'].set_color('#f0f0f0')

    # Set plot background to match app background
    fig_merged.patch.set_facecolor('#2c313d')
    ax_merged.set_facecolor('#2c313d')

    ax_merged.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    ax_merged.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    plt.xticks(rotation=45)
    
    st.pyplot(fig_merged)
else:
    st.warning("예측 결과 시각화를 위한 'merged_test.csv' 데이터를 불러오거나 처리할 수 없습니다.")