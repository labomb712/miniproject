import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import platform
from matplotlib import font_manager, rc
import requests # TMDB API 호출을 위해 추가
import datetime # 날짜 선택을 위해 추가

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
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

st.set_page_config(page_title="영화 예측 시스템", layout="centered")
st.title("🎬 영화 예측 시스템")
st.markdown("---")

# --- 2. 데이터 및 API 관련 함수 ---

@st.cache_data(show_spinner="영화 데이터를 불러오는 중입니다...")
def load_data(file_path):
    """
    CSV 파일에서 영화 데이터를 로드하고 기본 전처리를 수행합니다.
    """
    if not os.path.exists(file_path):
        st.error(f"Error: 데이터 파일 '{file_path}'을(를) 찾을 수 없습니다. 'data' 폴더에 파일을 넣어주세요.")
        st.stop()

    df = pd.read_csv(file_path)
    for col in ['누적관객수', '누적매출액']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['개봉일'] = pd.to_datetime(df['개봉일'], errors='coerce', format='%Y-%m-%d')
    
    df.dropna(subset=['누적관객수', '누적매출액', '개봉일', '감독', '장르', '제작국가'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    df['개봉_월'] = df['개봉일'].dt.month
    df['개봉_년'] = df['개봉일'].dt.year
    df['text_for_tfidf'] = df[['감독', '제작국가', '장르']].astype(str).agg(' '.join, axis=1)
    df['text_for_kobert'] = df.apply(
        lambda row: f"{row['감독']} 감독이 제작한 {row['제작국가']} 영화. 장르는 {row['장르']}이며, {row['개봉_년']}년 {row['개봉_월']}월에 개봉했습니다.",
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

# 유사도 행렬 계산
cosine_sim_tfidf = get_tfidf_similarity_matrix(df)
cosine_sim_kobert = get_kobert_similarity_matrix(df)

def get_recommendations(title, similarity_matrix, top_n=5):
    """
    선택된 영화와 유사한 영화를 추천합니다.
    """
    idx = title_to_index.get(title)
    if idx is None: 
        st.warning(f"'{title}'에 대한 인덱스를 찾을 수 없습니다. 추천할 수 없습니다.")
        return None
    
    if idx >= len(similarity_matrix):
        st.error(f"'{title}'에 대한 인덱스를 찾았으나({idx}), 추천 모델의 범위를 벗어납니다. 데이터를 다시 확인해주세요.")
        return None

    sim_scores = sorted(list(enumerate(similarity_matrix[idx])), key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    
    recommended_df = df.iloc[movie_indices][['영화명', '감독', '장르', '개봉일']].copy()
    recommended_df['포스터'] = recommended_df['영화명'].apply(get_movie_poster_url)
    return recommended_df[['포스터', '영화명', '감독', '장르', '개봉일']]

# --- 사이드바 추가 ---
st.sidebar.header("🔍 영화 검색 및 필터")

# 감독 필터
all_directors = ['전체 감독'] + sorted(df['감독'].unique().tolist())
selected_director = st.sidebar.selectbox("감독:", all_directors)

# 장르 필터
all_genres = ['전체 장르'] + sorted(df['장르'].unique().tolist())
selected_genre = st.sidebar.selectbox("장르:", all_genres)

# 개봉일 범위 검색
st.sidebar.markdown("---")
st.sidebar.subheader("개봉일 범위")
# 데이터프레임의 최소/최대 개봉일을 기준으로 기본값 설정
min_date_data = df['개봉일'].min().date() if not df.empty else datetime.date(2000, 1, 1)
max_date_data = df['개봉일'].max().date() if not df.empty else datetime.date.today()

start_date = st.sidebar.date_input("개봉일:", value=min_date_data, min_value=min_date_data, max_value=max_date_data)
end_date = st.sidebar.date_input("-------------------------", value=max_date_data, min_value=min_date_data, max_value=max_date_data)

# 날짜 유효성 검사
if start_date > end_date:
    st.sidebar.error("시작 개봉일은 종료 개봉일보다 빠를 수 없습니다.")
    # 유효하지 않은 경우 필터링을 하지 않도록 처리하거나, 기본값으로 되돌릴 수 있음
    date_filter_valid = False
else:
    date_filter_valid = True


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

# 필터링된 영화 목록이 비어있을 경우 처리
if filtered_df.empty and (selected_director != '전체 감독' or selected_genre != '전체 장르' or not date_filter_valid):
    st.warning("선택하신 조건에 해당하는 영화를 찾을 수 없습니다. 필터를 초기화하거나 다른 조건을 시도해보세요.")
    movie_list = ['영화를 선택하세요...'] # 드롭다운 초기화
    selected_movie = '영화를 선택하세요...' # 선택된 영화도 초기화
elif not filtered_df.empty:
    movie_list = ['영화를 선택하세요...'] + sorted(filtered_df['영화명'].unique().tolist())
    # 기존 선택된 영화가 필터링된 목록에 없으면 '영화를 선택하세요...'로 초기화
    if 'selected_movie' not in st.session_state or st.session_state.selected_movie not in movie_list:
        selected_movie = '영화를 선택하세요...'
    else:
        selected_movie = st.session_state.selected_movie
else: # 필터링 조건이 없을 경우 전체 영화 목록 사용
    movie_list = ['영화를 선택하세요...'] + sorted(df['영화명'].unique().tolist())
    if 'selected_movie' not in st.session_state:
        selected_movie = '영화를 선택하세요...'
    else:
        selected_movie = st.session_state.selected_movie


# 사이드바에서 필터링된 목록을 기반으로 영화 선택 드롭다운 생성 (메인 화면)
st.markdown("<strong>추천의 기준이 될 영화를 선택해주세요:</strong>", unsafe_allow_html=True)
selected_movie = st.selectbox("", movie_list, key="main_movie_selector", label_visibility="collapsed") 
# 선택된 영화를 session_state에 저장하여 필터 변경 시에도 유지되도록 함
st.session_state.selected_movie = selected_movie

# --- 4. Streamlit UI - 영화 추천 ---

st.header("✨ 콘텐츠 기반 영화 추천")

if selected_movie != '영화를 선택하세요...':
    movie_info_rows = df[df['영화명'] == selected_movie]
    
    if not movie_info_rows.empty:
        movie_info = movie_info_rows.iloc[0]
        
        # 선택된 영화 정보 (포스터와 함께)
        st.subheader(f"({selected_movie})정보")
        
        # 포스터와 정보 영역의 시각적 균형을 위한 고정된 컬럼 비율 설정
        # 이미지 크기가 커졌으므로, 정보 영역의 비율도 그에 맞춰 조절이 필요할 수 있습니다.
        # 여기서는 이미지 너비를 키웠으므로, col1과 col2의 비율은 다시 1:1에 가깝게 조정합니다.
        # 필요에 따라 [1, 1], [0.8, 1.2] 등 다시 시도해보세요.
        col1, col2 = st.columns([1, 2]) # 이미지 너비가 커졌으므로 컬럼 비율을 다시 조정
        
        with col1:
            # 이미지 너비를 400 픽셀로 설정 (원하는 픽셀 값으로 변경 가능)
            st.image(get_movie_poster_url(selected_movie), width=300) # 이미지 크기 키움
        with col2:
            # st.markdown을 사용하여 파란 배경을 제거하고 글자색을 기본(검은색)으로 설정
            # HTML <p> 태그와 style 속성을 사용하여 글씨 크기 키움
            st.markdown(f"<p style='font-size:31px;'><strong>감독:</strong> {movie_info['감독']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:24px;'><strong>장르:</strong> {movie_info['장르']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:24px;'><strong>제작국가:</strong> {movie_info['제작국가']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:24px;'><strong>개봉일:</strong> {movie_info['개봉일'].date()}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:24px;'><strong>누적 관객수:</strong> {int(movie_info['누적관객수']):,} 명</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:24px;'><strong>누적 매출액:</strong> ₩ {int(movie_info['누적매출액']):,}</p>", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader(f"({selected_movie})와 비슷한 영화 추천 목록")
        
        rec_col1, rec_col2 = st.columns(2)
        with rec_col1:
            st.markdown("<p style='font-size:25px;'><strong>🤖 TF-IDF기반 추천 (키워드 중심)</strong></p>", unsafe_allow_html=True)
            rec_tfidf = get_recommendations(selected_movie, cosine_sim_tfidf)
            if rec_tfidf is not None and not rec_tfidf.empty:
                # 추천 테이블 내 포스터 크기도 작게 유지 (선택사항)
                st.data_editor(rec_tfidf, column_config={"포스터": st.column_config.ImageColumn("포스터", width="small")}, hide_index=True, use_container_width=True)
            else:
                st.warning("TF-IDF 기반 추천 결과를 찾을 수 없습니다.")
        with rec_col2:
            st.markdown("<p style='font-size:25px;'><strong>🧠 KoBERT기반 추천 (의미 중심)</strong></p>", unsafe_allow_html=True)
            rec_kobert = get_recommendations(selected_movie, cosine_sim_kobert)
            if rec_kobert is not None and not rec_kobert.empty:
                # 추천 테이블 내 포스터 크기도 작게 유지 (선택사항)
                st.data_editor(rec_kobert, column_config={"포스터": st.column_config.ImageColumn("포스터", width="small")}, hide_index=True, use_container_width=True)
            else:
                st.warning("KoBERT 기반 추천 결과를 찾을 수 없습니다.")
    else:
        st.error(f"선택한 영화 '{selected_movie}'의 정보를 데이터에서 찾을 수 없습니다.")


st.markdown("\n\n---\n\n")

# --- 5. Streamlit UI - 누적 관객수 예측 ---

st.header("🎯 누적 관객수 예측 모델")
with st.spinner("관객수 예측 모델을 학습하는 중입니다..."):
    if df.empty or len(df) < 2:
        st.warning("모델 학습을 위한 데이터가 충분하지 않습니다. 파일과 데이터 전처리 결과를 확인해주세요.")
        mse, rmse, r2 = 0, 0, 0
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "데이터 부족으로 예측 불가", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16, color='gray')
        ax.axis('off')
    else:
        X = df[['누적매출액', '개봉_월', '장르', '제작국가']]
        y = df['누적관객수']
        numerical_features = ['누적매출액']
        categorical_features = ['개봉_월', '장르', '제작국가']
        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)],
            remainder='passthrough')
        model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
        
        test_size_val = 0.2 
        if len(X) < 10:
             test_size_val = max(0.2, 1 / len(X) if len(X) > 0 else 0.2) 

        try:
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
        except ValueError as e:
            st.error(f"데이터 분할 또는 모델 학습 중 오류 발생: {e}. 데이터셋 크기 또는 특성을 확인해주세요.")
            mse, rmse, r2 = 0, 0, 0
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "모델 학습 중 오류 발생", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16, color='red')
            ax.axis('off')
    st.pyplot(fig)