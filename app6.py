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
    
    # '누적관객수', '누적매출액'을 숫자형으로 변환 (변환 불가 시 NaN)
    for col in ['누적관객수', '누적매출액']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # '개봉일'을 datetime 객체로 변환 (변환 불가 시 NaT) - 파일 포맷에 맞춰 '%Y-%m-%d' 명시
    df['개봉일'] = pd.to_datetime(df['개봉일'], errors='coerce', format='%Y-%m-%d') 
    
    # 누락된 데이터로 인해 최신 날짜가 제거되는 것을 방지하기 위해 범주형 컬럼의 결측치 처리
    df['감독이름'].fillna('알 수 없음', inplace=True)
    df['제작국가'].fillna('알 수 없음', inplace=True)
    df['장르'].fillna('알 수 없음', inplace=True)

    # '누적관객수'와 '누적매출액'의 NaN 값은 0으로 채움 (모델 학습을 위해)
    df['누적관객수'].fillna(0, inplace=True)
    df['누적매출액'].fillna(0, inplace=True)

    # '개봉일'이 NaT인 행 (즉, 날짜로 변환되지 못한 값)과 '제목'이 없는 행 제거
    df.dropna(subset=['개봉일', '제목'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # --- 수정된 부분: 시작일을 2003년 10월 2일로 고정 ---
    fixed_start_date_data_filter = datetime.date(2003, 10, 2) 
    df = df[df['개봉일'].dt.date >= fixed_start_date_data_filter].copy() # 필터링 후 복사본 생성
    df.reset_index(drop=True, inplace=True) # 인덱스 재설정
    # --- 수정된 부분 끝 ---

    # --- 개발자용 디버깅 정보 제거 ---
    # st.sidebar.subheader("데이터 로드 디버깅 (개발자용)")
    # if not df.empty:
    #     st.sidebar.write(f"최종 DF 내 최소 개봉일: {df['개봉일'].min().date()}")
    #     st.sidebar.write(f"최종 DF 내 최대 개봉일: {df['개봉일'].max().date()}")
    #     st.sidebar.write("최소 개봉일 주변 데이터 (상위 5개):")
    #     st.sidebar.dataframe(df.sort_values(by='개봉일').head(5))
    # else:
    #     st.sidebar.write("최종 데이터프레임이 비어 있습니다.")
    # st.sidebar.markdown("---")
    # --- 개발자용 디버깅 정보 제거 끝 ---


    # 날짜 파생 특성 생성 (XGBoost 모델에 맞춤)
    df['개봉년도'] = df['개봉일'].dt.year
    df['개봉월'] = df['개봉일'].dt.month
    df['개봉요일'] = df['개봉일'].dt.weekday # 월요일=0, 일요일=6

    # 파생된 날짜 특성에 혹시 모를 NaN이 있다면 0으로 채움 (dropna 이후에는 거의 없을 것임)
    df['개봉년도'] = df['개봉년도'].fillna(0).astype(int)
    df['개봉월'] = df['개봉월'].fillna(0).astype(int)
    df['개봉요일'] = df['개봉요일'].fillna(0).astype(int)
    
    # 추천 모델을 위한 텍스트 특성
    df['text_for_tfidf'] = df[['감독이름', '제작국가', '장르']].astype(str).agg(' '.join, axis=1)
    df['text_for_kobert'] = df.apply(
        lambda row: f"{row['감독이름']} 감독이 제작한 {row['제작국가']} 영화. 장르는 {row['장르']}이며, {row['개봉년도']}년 {row['개봉월']}월에 개봉했습니다.",
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
DATA_FILE_PATH = "data/박스오피스_2003-11_2025-07_최종.csv" 
df = load_data(DATA_FILE_PATH)

# 데이터가 비어있을 경우 Early Exit
if df.empty:
    st.error("데이터 로드 및 전처리 후 데이터가 비어 있습니다. 파일 내용과 전처리 조건을 확인해주세요.")
    st.stop()

# '영화명' 대신 '제목' 컬럼 사용
title_to_index = pd.Series(df.index, index=df['제목']).drop_duplicates() 

# --- 3. 추천 모델 (TF-IDF & KoBERT) ---

@st.cache_resource(show_spinner="TF-IDF 유사도 모델을 계산하는 중입니다...")
def get_tfidf_similarity_matrix(dataframe):
    tfidf = TfidfVectorizer(min_df=2)
    tfidf_matrix = tfidf.fit_transform(dataframe['text_for_tfidf'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

@st.cache_resource(show_spinner="KoBERT 임베딩 및 유사도 모델을 계산하는 중입니다...")
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

    # 각 모델의 유사도 점수 가져오기
    scores_tfidf = sim_matrix_tfidf[idx]
    scores_kobert = sim_matrix_kobert[idx]

    # 가중치 합산
    combined_scores = (scores_tfidf * weight_tfidf) + (scores_kobert * weight_kobert)

    # 자기 자신 제외하고 유사도 점수 추출 및 정렬
    sim_scores = sorted(list(enumerate(combined_scores)), key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    
    recommended_df = df.iloc[movie_indices][['제목', '감독이름', '장르', '개봉일']].copy()
    recommended_df['포스터'] = recommended_df['제목'].apply(get_movie_poster_url)
    return recommended_df[['포스터', '제목', '감독이름', '장르', '개봉일']]


# --- 사이드바 추가 ---
st.sidebar.header("🔍 영화 검색 및 필터")

# 감독 필터
all_directors = ['전체 감독'] + sorted(df['감독이름'].unique().tolist())
selected_director = st.sidebar.selectbox("감독:", all_directors)

# 장르 필터
all_genres = ['전체 장르'] + sorted(df['장르'].unique().tolist())
selected_genre = st.sidebar.selectbox("장르:", all_genres)

# 개봉일 범위 검색
st.sidebar.markdown("---")
st.sidebar.subheader("개봉일 범위")

fixed_start_date_for_ui = datetime.date(2003, 10, 2) 
max_date_for_display = df['개봉일'].max().date() if not df.empty else datetime.date.today()

start_date = st.sidebar.date_input(
    "시작일:", 
    value=fixed_start_date_for_ui, 
    min_value=fixed_start_date_for_ui, 
    max_value=max_date_for_display, 
    key="sidebar_start_date"
)
end_date = st.sidebar.date_input(
    "종료일:", 
    value=max_date_for_display, 
    min_value=fixed_start_date_for_ui, 
    max_value=max_date_for_display, 
    key="sidebar_end_date"
)

# 날짜 유효성 검사
date_filter_valid = True
if start_date > end_date:
    st.sidebar.error("시작 개봉일은 종료 개봉일보다 빠를 수 없습니다.")
    date_filter_valid = False


# 필터링된 영화 목록 생성
filtered_df = df.copy()

if selected_director != '전체 감독':
    filtered_df = filtered_df[filtered_df['감독이름'] == selected_director] 

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
    movie_list = ['영화를 선택하세요...']
    selected_movie = '영화를 선택하세요...'
elif not filtered_df.empty:
    movie_list = ['영화를 선택하세요...'] + sorted(filtered_df['제목'].unique().tolist()) 
    if 'selected_movie' not in st.session_state or st.session_state.selected_movie not in movie_list:
        selected_movie = '영화를 선택하세요...'
    else:
        selected_movie = st.session_state.selected_movie
else: # 필터링 조건이 없을 경우 전체 영화 목록 사용
    movie_list = ['영화를 선택하세요...'] + sorted(df['제목'].unique().tolist()) 
    if 'selected_movie' not in st.session_state:
        selected_movie = '영화를 선택하세요...'
    else:
        selected_movie = st.session_state.selected_movie


st.markdown("<strong>추천의 기준이 될 영화를 선택해주세요:</strong>", unsafe_allow_html=True)
selected_movie = st.selectbox("", movie_list, key="main_movie_selector", label_visibility="collapsed") 
st.session_state.selected_movie = selected_movie

# --- 4. Streamlit UI - 영화 추천 ---

st.header("✨ 콘텐츠 기반 영화 추천")
st.write("영화를 선택하면 해당 영화의 포스터와 정보, 그리고 융합된 방식으로 추천된 영화 목록을 보여줍니다.")

if selected_movie != '영화를 선택하세요...':
    st.markdown("---") 
    movie_info_rows = df[df['제목'] == selected_movie] 
    
    if not movie_info_rows.empty:
        movie_info = movie_info_rows.iloc[0]
        
        st.subheader(f"[{selected_movie}] 정보")
        
        col1, col2 = st.columns([1, 2]) 
        
        with col1:
            st.image(get_movie_poster_url(selected_movie), width=300) 
        with col2:
            st.markdown(f"<p style='font-size:31px;'><strong>감독:</strong> {movie_info['감독이름']}</p>", unsafe_allow_html=True) 
            st.markdown(f"<p style='font-size:24px;'><strong>장르:</strong> {movie_info['장르']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:24px;'><strong>제작국가:</b> {movie_info['제작국가']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:24px;'><strong>개봉일:</strong> {movie_info['개봉일'].date()}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:24px;'><strong>누적 관객수:</strong> {int(movie_info['누적관객수']):,} 명</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:24px;'><strong>누적 매출액:</strong> ₩ {int(movie_info['누적매출액']):,}</p>", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader(f"[{selected_movie}]와 비슷한 영화 추천 목록")
        
        # 사이드바에 가중치 조절 라디오 버튼 추가
        st.sidebar.markdown("---")
        st.sidebar.subheader("추천 기준")
        
        recommendation_mode = st.sidebar.radio(
            "어떤 기준으로 추천하시겠어요?",
            ('의미 중심', '중간', '키워드 중심'),
            index=1, 
            key="recommendation_mode"
        )

        weight_tfidf = 0.5 
        if recommendation_mode == '의미 중심':
            weight_tfidf = 0.0
        elif recommendation_mode == '키워드 중심':
            weight_tfidf = 1.0
        
        weight_kobert = 1.0 - weight_tfidf 
        
        st.markdown("<p style='font-size:25px;'><b>✨ 추천영화</b></p>", unsafe_allow_html=True)
        # 병합된 추천 모델 사용
        rec_combined = get_combined_recommendations(
            selected_movie, 
            cosine_sim_tfidf, 
            cosine_sim_kobert, 
            top_n=5, 
            weight_tfidf=weight_tfidf, 
            weight_kobert=weight_kobert
        )
        if rec_combined is not None and not rec_combined.empty:
            st.data_editor(rec_combined, column_config={"포스터": st.column_config.ImageColumn("포스터", width="small")}, hide_index=True, use_container_width=True)
        else:
            st.warning("융합 추천 결과를 찾을 수 없습니다.")
    else:
        st.error(f"선택한 영화 '{selected_movie}'의 정보를 데이터에서 찾을 수 없습니다.")


st.markdown("\n\n---\n\n")

# --- 5. Streamlit UI - 누적 관객수 예측 (XGBoost 모델) ---

st.header("🎯 누적 관객수 예측 모델")
with st.spinner("관객수 예측 모델을 학습하는 중입니다..."):
    # XGBoost 모델의 특성 컬럼 정의
    xgb_features = ['감독이름', '제작국가', '장르', '개봉년도', '개봉월', '개봉요일', '누적매출액']
    xgb_target = '누적관객수'

    # 필요한 모든 컬럼이 DataFrame에 있는지 최종 확인
    required_for_xgb = xgb_features + [xgb_target]
    if not all(col in df.columns for col in required_for_xgb):
        missing_cols = [col for col in required_for_xgb if col not in df.columns]
        st.error(f"XGBoost 모델 학습에 필요한 다음 컬럼이 없습니다: {', '.join(missing_cols)}. 데이터 파일을 확인해주세요.")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "필수 데이터 컬럼 누락", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16, color='red')
        ax.axis('off')
        st.pyplot(fig)
        st.stop() 

    # XGBoost 모델 학습 전 데이터 준비
    xgb_df = df.copy()

    # 범주형 컬럼 Label Encoding (XGBoost 모델용)
    le = LabelEncoder()
    for col in ['감독이름', '제작국가', '장르']:
        xgb_df[col] = le.fit_transform(xgb_df[col].astype(str)) 

    X_xgb = xgb_df[xgb_features]
    y_xgb = xgb_df[xgb_target]

    if X_xgb.empty or len(X_xgb) < 2:
        st.warning("XGBoost 모델 학습을 위한 데이터가 충분하지 않습니다. 파일 내용과 전처리 결과를 확인해주세요.")
        mse, rmse, r2 = 0, 0, 0
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "데이터 부족으로 예측 불가", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16, color='gray')
        ax.axis('off')
    else:
        try:
            # 데이터 분할
            X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
                X_xgb, y_xgb, test_size=0.2, random_state=42
            )

            # XGBoost DMatrix 생성
            dtrain_xgb = xgb.DMatrix(X_train_xgb, label=y_train_xgb)
            dtest_xgb = xgb.DMatrix(X_test_xgb, label=y_test_xgb)

            # XGBoost 모델 파라미터
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

            # XGBoost 모델 훈련
            model_xgb = xgb.train(
                params_xgb,
                dtrain_xgb,
                num_boost_round=1000,
                evals=[(dtrain_xgb, 'train'), (dtest_xgb, 'valid')],
                early_stopping_rounds=50,
                verbose_eval=False 
            )

            # 예측 수행
            y_pred_xgb = model_xgb.predict(dtest_xgb)
            y_pred_xgb[y_pred_xgb < 0] = 0 

            # 모델 성능 지표 계산
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

            st.subheader("📈 실제 vs 예측 관객수 시각화")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=y_test_xgb, y=y_pred_xgb, alpha=0.6, ax=ax, color='royalblue')
            ax.plot([y_test_xgb.min(), y_test_xgb.max()], [y_test_xgb.min(), y_test_xgb.max()], 'r--', lw=2, label='이상적인 예측')
            ax.set_xlabel("실제 누적 관객수")
            ax.set_ylabel("예측 누적 관객수")
            ax.set_title("XGBoost 회귀: 실제 vs 예측")
            ax.legend()
            ax.grid(True)
            ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
            ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
            plt.xticks(rotation=45)
        except Exception as e: 
            st.error(f"XGBoost 모델 학습 또는 예측 중 오류 발생: {e}. 데이터셋 크기 또는 특성을 확인해주세요.")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "모델 학습 중 오류 발생", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16, color='red')
            ax.axis('off')
    st.pyplot(fig)