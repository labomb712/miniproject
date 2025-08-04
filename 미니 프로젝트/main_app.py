import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime

# 다른 파일에서 정의된 함수/클래스 임포트
from utils.data_loader import load_data, load_and_preprocess_merged_test_data
from utils.api_utils import get_movie_poster_url
from utils.plot_utils import setup_korean_font, apply_chart_theme
from models.recommendation_model import get_tfidf_similarity_matrix, get_kobert_similarity_matrix, get_combined_recommendations
from models.prediction_model import train_and_predict_xgboost_model, calculate_audience_benchmark

# --- 0. 기본 설정 및 CSS 적용 ---
setup_korean_font()
st.set_page_config(page_title="영화 예측 시스템", layout="centered", initial_sidebar_state="expanded")

# 사용자 정의 CSS
st.markdown("""
<style>
    /* 전체 앱 배경 */
    .stApp {
        background-color: #F5F7FA; /* 배경색: 매우 연한 푸른빛 회색 */
        color: #1F2937; /* 텍스트색: 아주 어두운 회색 */
    }

/* 메인 타이틀 스타일 */
.stApp {
        background-color: #F5F7FA; /* 배경색: 매우 연한 푸른빛 회색 */
        color: #1F2937; /* 텍스트색: 아주 어두운 회색 */
}

    /* 사이드바 (네비게이션 바) 배경 */
    .stSidebar {
    background-color: #64748B;
    border-right: 1px solid #C5D9FA;
    width: 560vw; /* 뷰포트 너비의 20% */
    min-width: 200px; /* 최소 너비 */
    max-width: 600px; /* 최대 너비 (선택 사항) */
}            
    /* 사이드바 헤더 및 일반 텍스트 색상 */
    /* !important를 추가하여 우선순위 강제 적용 */
    .stSidebar [data-testid="stSidebarHeader"] h2,
    .stSidebar [data-testid="stSidebarHeader"] h3,
    .stSidebar [data-testid="stSidebarHeader"] h4,
    .stSidebar [data-testid="stSidebarHeader"] h5,
    .stSidebar [data-testid="stSidebarHeader"] h6,
    .stSidebar [data-testid="stText"] p, /* st.write, st.markdown 등 일반 텍스트 */
    .stSidebar [data-testid="stMarkdown"] p, /* st.markdown으로 생성된 텍스트 */
    .stSidebar [data-testid="stMetricLabel"], /* st.metric 라벨 */
    .stSidebar label, /* 모든 라벨 */
    /* 사이드바 내부의 st.header와 st.subheader 텍스트 */
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar h5, .stSidebar h6 { 
        color: #FFFFFF !important; /* 사이드바 텍스트색: 흰색으로 변경 */
    }
            
    /* 특히 selectbox, dateinput 등 위젯의 라벨 텍스트에 직접 접근 */
    .stSidebar [data-testid="stSelectbox"] label,
    .stSidebar [data-testid="stDateInput"] label,
    .stSidebar [data-testid="stSlider"] label,
    .stSidebar [data-testid="stTextInput"] label {
        color: #FFFFFF !important; /* 사이드바 텍스트색: 흰색으로 변경 */
    }
            
     /* 사이드바 내의 수평선 (---) 색상 변경 */
    .stSidebar hr {
        border-top: 1px solid #FFFFFF !important; /* 선의 색상을 흰색으로 강제 변경 */
    }
            
    /* --- 일반 버튼 스타일 (추천 기준 조정에 사용) --- */
    /* 기본 버튼 스타일 */
    .stButton > button {
        background-color: #D1D5DB; /* 연한 회색 */
        color: #1F2937; /* 어두운 텍스트 */
        border: 1px solid #9CA3AF; /* 중간 회색 테두리 */
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.2s ease-in-out;
        width: 100%; /* 컬럼 내에서 가득 채우도록 */
        cursor: pointer;
    }
            
    /* 버튼 호버 시 스타일 */
    .stButton > button:hover {
        background-color: #BFDBFE; /* 강조색보다 연한 파랑 */
        color: #2563EB; /* 강조색 */
        border-color: #2563EB; /* 강조색 */
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
            
    /* 선택된 버튼 스타일 (JS를 통해 클래스 추가) */
    .stButton > button.selected-button {
        background-color: #2563EB !important; /* 강조색 */
        color: #FFFFFF !important; /* 흰색 텍스트 */
        border-color: #2563EB !important; /* 강조색 */
        box-shadow: 0 0 10px rgba(37, 99, 235, 0.5); /* 강조색 그림자 */
    }
            
    /* 선택되지 않은 버튼 스타일 */
    .stButton > button.unselected-button {
        background-color: #D1D5DB; /* 연한 회색 */
        color: #1F2937; /* 어두운 텍스트 */
        border: 1px solid #9CA3AF; /* 중간 회색 테두리 */
    }

    /* Streamlit 컬럼 내 버튼의 불필요한 마진 제거 및 정렬 */
    div[data-testid="column"] > div > .stButton {
        margin-bottom: 0px; /* 버튼 하단 마진 제거 */
    }

    /* 메인 컨텐츠 헤더 및 타이틀 색상 */
    h2, h3, h4, h5, h6 {
        color: #64748B; /* 강조색 */
    }
    
    /* h1 색상을 검은색으로 변경 (명시적으로) */
    h1 {
        color: #1F2937 !important; /* 텍스트색 */
    }
    
    /* 메인 타이틀 설명 텍스트 */
    .stMarkdown p {
        color: #1F2937; /* 텍스트색 */
    }
            
    /* selectbox, textinput, dateinput 등 기타 위젯 배경 및 텍스트 색상 */
    .stSelectbox > div > div, .stTextInput > div > div > input, .stDateInput > div > div > input {
        background-color: #FFFFFF; /* 흰색 배경 */
        color: #1F2937; /* 텍스트색 */
        border: 1px solid #D1D5DB; /* 연한 회색 테두리 */
        border-radius: 5px;
    }
    
    /* Selectbox 드롭다운 아이템 */
    .stSelectbox div[role="listbox"] {
        background-color: #FFFFFF;
        color: #1F2937;
    }
    .stSelectbox div[role="option"] {
        color: #1F2937;
    }
    .stSelectbox div[role="option"]:hover {
        background-color: #BFDBFE; /* 강조색보다 연한 파랑 */
        color: #2563EB; /* 강조색 */
    }

    /* Metric 카드 */
    [data-testid="stMetric"] {
        background-color: #FFFFFF; /* 흰색 배경 */
        border: 1px solid #D1D5DB; /* 연한 회색 테두리 */
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    [data-testid="stMetricLabel"] {
        color: #1F2937; /* 텍스트색 */
    }
    [data-testid="stMetricValue"] {
        color: #2563EB; /* 강조색 */
        font-size: 2em;
        font-weight: bold;
    }
    [data-testid="stMetricDelta"] {
        color: #2563EB; /* 강조색 */
    }

    /* 경고/에러 메시지 */
    .stAlert {
        background-color: #DBEAFE; /* 연한 파랑 배경 */
        color: #1E40AF; /* 진한 파랑 텍스트 */
        border-left: 5px solid #2563EB; /* 강조색 하이라이트 */
        border-radius: 5px;
    }
    .stWarning {
        background-color: #FEF3C7; /* 연한 노랑 배경 */
        color: #92400E; /* 진한 노랑 텍스트 */
        border-left: 5px solid #FBBF24; /* 노랑 하이라이트 */
        border-radius: 5px;
    }
            
    
    /* 새로운 추천 목록 스타일 */
    .recommendation-list-item {
        display: flex;
        align-items: center;
        background-color: #FFFFFF; /* 흰색 배경 */
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border-left: 5px solid #2563EB; /* 강조색 하이라이트 */
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }

    .recommendation-list-item:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(37, 99, 235, 0.25); /* 강조색 그림자 */
    }

    .recommendation-list-item img {
        width: 80px;
        height: 120px;
        object-fit: cover;
        border-radius: 5px;
        margin-right: 15px;
        flex-shrink: 0;
        border: 1px solid #D1D5DB; /* 연한 회색 테두리 */
    }

    .movie-info-container {
        display: flex;
        flex-direction: column;
        text-align: left;
        flex-grow: 1;
    }

    .movie-title-list {
        font-size: 18px;
        color: #2563EB; /* 강조색 */
        font-weight: bold;
        margin-bottom: 5px;
        line-height: 1.3;
    }

    .movie-detail-list {
        font-size: 14px;
        color: #1F2937; /* 텍스트색 */
        margin-bottom: 3px;
        line-height: 1.3;
    }
    .movie-detail-list strong {
        color: #1F2937; /* 텍스트색 */
    }
    /* 마지막 항목의 하단 마진 제거 */
    .movie-info-container .movie-detail-list:last-of-type {
        margin-bottom: 0;
    }


    /* Matplotlib plot 배경을 Streamlit 앱 배경과 동일하게 설정 */
    .stPlotlyChart {
        background-color: #FFFFFF; /* 흰색 배경 */
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
            
    /* 영화 상세 정보 창 조절 */
    .movie-detail-box {
        background-color: #FFFFFF; /* 흰색 배경 */
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .movie-detail-item {
        margin-bottom: 8px;
        font-size: 22px !important;
        font-weight: bold !important;
        line-height: 1.4;
        color: #1F2937; /* 텍스트색 */
    }
    .movie-detail-item strong {
        color: #2563EB; /* 강조색 */
    }

    /* Streamlit header (h1) 위에 추가된 여백 제거 */
    h1 {
        padding-top: 0rem;
    }

    /* Streamlit 제목 영역 전체를 아우르는 스타일 */
    .stDeck {
    background-color: #F5F7FA; /* 배경색: 매우 연한 푸른빛 회색 */
    padding: 1rem 0; /* 위아래로 1rem 패딩, 좌우 패딩 없음 */
    margin-bottom: 1rem; /* 아래쪽 여백 */
    border-bottom: 1px solid #D1D5DB; /* 연한 회색 하단 라인 */
    }
            
    /* 추천 기준 조정 버튼 (KoBERT, TF-IDF) */
    .stButton > button {
    background-color: #D1D5DB; /* 기본 버튼 배경: 연한 회색 */
    color: #1F2937; /* 기본 버튼 텍스트: 어두운 텍스트 */
    border: 1px solid #9CA3AF; /* 중간 회색 테두리 */
    border-radius: 5px; /* 둥근 모서리 */
    padding: 10px 20px; /* 내부 여백 */
    font-weight: bold; /* 글씨 굵게 */
    transition: all 0.2s ease-in-out; /* 모든 속성 변경 시 0.2초간 부드러운 전환 */
    width: 100%; /* 컬럼 내에서 가득 채우도록 */
    cursor: pointer; /* 마우스 오버 시 손가락 커서 */
    }

    .stButton > button:hover {
    background-color: #BFDBFE; /* 버튼 호버 시 배경: 강조색보다 연한 파랑 */
    color: #2563EB; /* 버튼 호버 시 텍스트: 강조색 (진한 파랑) */
    border-color: #2563EB; /* 버튼 호버 시 테두리: 강조색 (진한 파랑) */
    transform: translateY(-2px); /* 호버 시 버튼을 살짝 위로 이동 */
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* 호버 시 그림자 추가 */
    }

    .selected-button {
    background-color: #2563EB !important; /* 선택된 버튼 배경: 강조색 (진한 파랑) */
    color: #FFFFFF !important; /* 선택된 버튼 텍스트: 흰색 */
    border-color: #2563EB !important; /* 선택된 버튼 테두리: 강조색 (진한 파랑) */
    box-shadow: 0 0 10px rgba(37, 99, 235, 0.5); /* 강조색 그림자 */
    }

    .unselected-button {
    background-color: #D1D5DB !important; /* 선택되지 않은 버튼 배경: 연한 회색 */
    color: #1F2937 !important; /* 선택되지 않은 버튼 텍스트: 어두운 텍스트 */
    border: 1px solid #9CA3AF !important; /* 중간 회색 테두리 */
    }

    /* 예측 카드 스타일 */
    .prediction-card {
    background-color: #FFFFFF; /* 배경: 흰색 */
    padding: 1.5em; /* 내부 여백 */
    border-radius: 10px; /* 둥근 모서리 */
    text-align: center; /* 텍스트 중앙 정렬 */
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* 연한 그림자 */
    border: 1px solid #E5E7EB; /* 아주 연한 회색 테두리 */
    cursor: pointer; /* 마우스 오버 시 손가락 커서 */
    transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s; /* 부드러운 전환 효과 */
    height: 100%; /* 컬럼 내에서 높이 균일하게 */
    display: flex; /* Flexbox 사용 */
    flex-direction: column; /* 세로 방향 정렬 */
    justify-content: center; /* 세로 중앙 정렬 */
    }

    .prediction-card:hover {
    transform: translateY(-5px); /* 호버 시 카드를 살짝 위로 이동 */
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2); /* 호버 시 그림자 진하게 */
    border-color: #2563EB; /* 호버 시 테두리색 강조색 (진한 파랑) */
    }

    .prediction-card h4 {
        color: #4B5563; /* 라벨 색상: 어두운 회색 */
        margin-bottom: 0.5em;
        font-size: 1.1em; /* 폰트 크기 조정 */
    }

    .prediction-card p {
        font-size: 2em; /* 숫자 폰트 크기: 크게 */
        font-weight: bold; /* 글씨 굵게 */
        color: #1F2937; /* 숫자 색상: 어두운 텍스트 */
        line-height: 1.2; /* 줄 간격 */
    }

    /* 숨겨진 버튼 스타일 (클릭 이벤트 트리거용) */
    [data-testid="stButton"] button[id^="trigger_"] {
    display: none !important; /* 해당 버튼을 화면에 표시하지 않음 */
    }

</style>
""", unsafe_allow_html=True)

st.markdown(f"""
    <div class="stDeck">
        <h1 style="text-align: center; color: #1F2937;">🎬 영화 예측 시스템:
        <br>당신의 다음 영화를 발견하세요!</h1>
        <p style="font-size:20px; text-align: center; color: #1F2937;">
        "영화는 우리에게 꿈을 꾸게 합니다." 🌟 
        <br>원하는 영화를 탐색하고, 새로운 추천을 받으며, 흥행 성과를 예측해보세요.
        </p>
    </div>
    """, unsafe_allow_html=True)
st.markdown("---")

# ✨ 세션 상태 초기화 (main_app.py에 유지)
if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = '영화를 선택하세요...'
if 'charted_movies' not in st.session_state:
    st.session_state.charted_movies = []
if 'display_chart_mode' not in st.session_state:
    st.session_state.display_chart_mode = 'audience'
if 'selected_compare_movie' not in st.session_state:
    st.session_state.selected_compare_movie = "비교할 영화를 선택하세요..."
if 'recommendation_mode' not in st.session_state:
    st.session_state.recommendation_mode = '장르 (KoBERT)'

# --- 1. 데이터 로드 ---
DATA_FILE_PATH = "data/청불제거_최종_DB컬럼.csv"
MERGED_TEST_FILE_PATH = "data/merged_test.csv"

df = load_data(DATA_FILE_PATH)
merged_test_df = load_and_preprocess_merged_test_data(MERGED_TEST_FILE_PATH)

if df.empty or merged_test_df.empty:
    st.error("필수 데이터 파일 로드에 실패했습니다. 앱을 실행할 수 없습니다.")
    st.stop()

# 영화명 -> 인덱스 매핑 (추천 시스템용)
title_to_index = pd.Series(df.index, index=df['영화명']).drop_duplicates()

# --- 2. 사이드바 (필터링) ---
st.sidebar.header("🔍 영화 검색 및 필터")
st.sidebar.markdown("---")

# 사이드바 필터링 로직 (현재 파일에 유지하거나 별도 모듈 함수로 분리 가능)
all_directors = ['전체 감독'] + sorted(df['감독'].unique().tolist())
selected_director = st.sidebar.selectbox("감독:", all_directors)

all_genres = ['전체 장르'] + sorted(df['장르'].unique().tolist())
selected_genre = st.sidebar.selectbox("장르:", all_genres)

st.sidebar.markdown("---")
st.sidebar.subheader("📅 개봉일 범위")

# 개봉일 범위 설정 (유효한 날짜가 없을 경우 기본값 설정)
min_date_for_display = df['개봉일'].min().date() if not df.empty and pd.notna(df['개봉일'].min()) else datetime.date(2000, 1, 1)
max_date_for_display = df['개봉일'].max().date() if not df.empty and pd.notna(df['개봉일'].max()) else datetime.date.today()

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

main_movie_list = ['영화를 선택하세요...'] + sorted(filtered_df['영화명'].unique().tolist()) 

def set_selected_movie_callback():
    st.session_state.selected_movie = st.session_state.main_movie_selector
    st.session_state.display_chart_mode = 'audience'
    # 선택된 영화가 차트 목록에 있으면 제거 (메인 영화는 항상 포함되므로 중복 방지)
    st.session_state.charted_movies = [m for m in st.session_state.charted_movies if m['영화명'] != st.session_state.selected_movie]

st.markdown("<strong><p style='font-size:22px;'>🔍 추천의 기준이 될 영화를 선택해주세요:</p></strong>", unsafe_allow_html=True)
st.selectbox(
    "",
    main_movie_list,
    key="main_movie_selector",
    label_visibility="collapsed",
    on_change=set_selected_movie_callback
)

# 사용자가 직접 선택하지 않았는데 필터링으로 인해 선택된 영화가 목록에 없는 경우 기본값으로 되돌림
if st.session_state.selected_movie not in main_movie_list:
    st.session_state.selected_movie = '영화를 선택하세요...'

# --- 3. 영화 추천 및 상세 정보 섹션 ---
st.header("✨ 콘텐츠 기반 영화 추천")
st.write("아래에서 영화를 선택하면 해당 영화의 상세 정보와 함께 비슷한 영화들을 추천해 드립니다.")

if st.session_state.selected_movie != '영화를 선택하세요...':
    movie_info_rows = df[df['영화명'] == st.session_state.selected_movie] 
    
    if not movie_info_rows.empty:
        movie_info = movie_info_rows.iloc[0]
        
        st.subheader(f"[{st.session_state.selected_movie}] 상세 정보")
        col1, col2 = st.columns([1, 2]) 
        
        with col1:
            st.image(get_movie_poster_url(st.session_state.selected_movie), width=200, caption=f"'{st.session_state.selected_movie}' 포스터") 
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
        st.subheader(f"[{st.session_state.selected_movie}]와 비슷한 영화 추천 목록 🍿")
        
        # --- 추천 기준 조정 ---
        st.subheader("⚖️ 추천 기준 조정")
        col_kobert, col_tfidf = st.columns(2)

        with col_kobert:
            kobert_button_label = '장르 (KoBERT)'
            if st.button(kobert_button_label, key="btn_kobert"):
                st.session_state.recommendation_mode = kobert_button_label

        with col_tfidf:
            tfidf_button_label = '감독 (TF-IDF)'
            if st.button(tfidf_button_label, key="btn_tfidf"):
                st.session_state.recommendation_mode = tfidf_button_label

        # 자바스크립트를 이용한 버튼 시각화 업데이트
        st.markdown(f"""
            <script>
                var kobertButton = document.querySelector('[data-testid="stButton"] button[key="btn_kobert"]');
                if (kobertButton) {{
                    if ("{st.session_state.recommendation_mode}" === "장르 (KoBERT)") {{
                        kobertButton.classList.add('selected-button');
                        kobertButton.classList.remove('unselected-button');
                    }} else {{
                        kobertButton.classList.add('unselected-button');
                        kobertButton.classList.remove('selected-button');
                    }}
                }}

                var tfidfButton = document.querySelector('[data-testid="stButton"] button[key="btn_tfidf"]');
                if (tfidfButton) {{
                    if ("{st.session_state.recommendation_mode}" === "감독 (TF-IDF)") {{
                        tfidfButton.classList.add('selected-button');
                        tfidfButton.classList.remove('unselected-button');
                    }} else {{
                        tfidfButton.classList.add('unselected-button');
                        tfidfButton.classList.remove('selected-button');
                    }}
                }}
            </script>
        """, unsafe_allow_html=True)
        # --- 추천 기준 조정 끝 ---

        weight_tfidf = 0.5 
        if st.session_state.recommendation_mode == '장르 (KoBERT)':
            weight_tfidf = 0.0 
        elif st.session_state.recommendation_mode == '감독 (TF-IDF)':
            weight_tfidf = 1.0 
        weight_kobert = 1.0 - weight_tfidf 
        
        st.markdown("<p style='font-size:20px;'><b>👍 당신을 위한 추천 영화들:</b></p>", unsafe_allow_html=True)

        cosine_sim_tfidf = get_tfidf_similarity_matrix(df)
        cosine_sim_kobert = get_kobert_similarity_matrix(df)

        # get_combined_recommendations 함수가 이제 인덱스만 반환하므로, 이를 바탕으로 DataFrame을 재구성합니다.
        recommended_indices = get_combined_recommendations(
            st.session_state.selected_movie, 
            cosine_sim_tfidf, 
            cosine_sim_kobert, 
            title_to_index,
            top_n=5, 
            weight_tfidf=weight_tfidf, 
            weight_kobert=weight_kobert
            # 'get_movie_poster_url' 인자 제거됨
        )

        # recommended_indices가 None 또는 비어있을 경우 처리
        if recommended_indices is None or not recommended_indices:
            rec_combined = pd.DataFrame() # 빈 DataFrame으로 초기화
        else:
            # 반환된 인덱스를 사용하여 df에서 영화 정보를 추출하여 새로운 DataFrame 생성
            rec_combined = df.iloc[recommended_indices][['영화명', '감독', '장르', '개봉일']].copy()
            rec_combined['포스터'] = rec_combined['영화명'].apply(get_movie_poster_url)
            rec_combined = rec_combined[['포스터', '영화명', '감독', '장르', '개봉일']] # 순서 재정렬

        def select_recommended_movie_for_detail(movie_name):
            st.session_state.selected_movie = movie_name

        if rec_combined is not None and not rec_combined.empty:
            for i, row in rec_combined.iterrows():
                # 고유한 key 생성을 위해 인덱스 i와 영화명을 함께 사용
                unique_key = f"rec_movie_{row['영화명'].replace(' ', '_')}_{i}" # 공백 제거하여 key 오류 방지
                st.markdown(
                    f"""
                    <div class="recommendation-list-item" onclick="
                        const button = document.querySelector('[data-testid="stButton"] button[id^=\\'trigger_{unique_key}_\\']');
                        if (button) button.click();
                    ">
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
                # 숨겨진 Streamlit 버튼을 추가하여 JavaScript 클릭 이벤트를 트리거
                st.button(
                    "Select",
                    key=f"trigger_{unique_key}_{st.session_state.get('last_rec_click_id', 0)}",
                    on_click=select_recommended_movie_for_detail,
                    args=(row['영화명'],),
                    help="이 버튼은 숨겨져 있으며, 추천 영화 카드를 클릭하면 동작합니다.",
                    type="secondary"
                )
                # 고유한 키를 위한 카운터 증가
                st.session_state.last_rec_click_id = st.session_state.get('last_rec_click_id', 0) + 1

            # 숨겨진 버튼에 대한 CSS
            st.markdown("""
                <style>
                    [data-testid="stButton"] button[id^="trigger_rec_movie_"] {
                        display: none !important;
                    }
                </style>
            """, unsafe_allow_html=True)
        else:
            st.warning("융합 추천 결과를 찾을 수 없습니다. 다른 영화를 선택하거나 데이터가 충분한지 확인해주세요.")
    else:
        st.error(f"선택한 영화 '{st.session_state.selected_movie}'의 정보를 데이터에서 찾을 수 없습니다.")


st.markdown("\n\n---\n\n")

# --- 4. 누적 관객수 예측 모델 (XGBoost) 섹션 ---
st.header("📈 누적 관객수 예측 모델 (XGBoost)")
st.write("이 섹션에서는 XGBoost 모델을 사용하여 영화의 누적 관객수를 예측하고, 모델의 성능을 시각화합니다.")

try:
    fig_xgb = train_and_predict_xgboost_model(df)
    st.pyplot(fig_xgb)
except Exception as e:
    st.error(f"XGBoost 모델 예측 섹션 로드 중 오류 발생: {e}")
    fig, ax = plt.subplots(figsize=(10, 6))
    apply_chart_theme(fig, ax)
    ax.text(0.5, 0.5, "예측 모델 로드 오류", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16, color='#FFD700')
    ax.axis('off')
    st.pyplot(fig)


st.markdown("\n\n---\n\n")

# --- 5. 통합된 '영화 예측 결과 및 비교' 섹션 ---
st.header("🔮 영화 예측 결과 및 비교 (CatBoost & XGBoost)") 
st.write("선택된 영화의 예상 누적 관객수와 평점을 확인하고, 최대 5개의 영화를 추가하여 비교 그래프를 그려보세요.")

if not merged_test_df.empty:
    audience_기준값_merged = calculate_audience_benchmark(merged_test_df)

    if st.session_state.selected_movie != '영화를 선택하세요...':
        selected_movie_data_for_display = merged_test_df[merged_test_df['영화명'] == st.session_state.selected_movie]

        if not selected_movie_data_for_display.empty:
            current_predicted_audience = selected_movie_data_for_display.iloc[0]['예측_누적관객수']
            current_actual_rating = selected_movie_data_for_display.iloc[0]['평점'] 

            st.markdown(f"#### '{st.session_state.selected_movie}'의 예상 성과")
            st.markdown("👇 아래 카드를 클릭하여 해당 영화의 자세한 그래프를 확인하세요.")

            col_audience_single, col_rating_single = st.columns(2)

            def set_display_chart_mode(chart_type):
                st.session_state.display_chart_mode = chart_type

            with col_audience_single:
                audience_card_class_single = "prediction-card"
                if st.session_state.display_chart_mode == 'audience':
                    audience_card_class_single += " selected"
                
                st.markdown(
                    f"""
                    <div class="{audience_card_class_single}" onclick="
                        const button = document.querySelector('[data-testid="stButton"] button[id^="trigger_chart_audience_"]');
                        if (button) button.click();
                    ">
                        <h4>예상 누적 관객수</h4>
                        <p>{int(current_predicted_audience):,} 명</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                st.button(
                    "예상 누적 관객수", 
                    key=f"trigger_chart_audience_{st.session_state.get('chart_audience_click_counter', 0)}", 
                    on_click=set_display_chart_mode, args=('audience',),
                    help="이 버튼은 숨겨져 있으며, 위 카드를 클릭하면 동작합니다.",
                    type="secondary"
                )
                st.session_state.chart_audience_click_counter = st.session_state.get('chart_audience_click_counter', 0) + 1


            with col_rating_single:
                rating_card_class_single = "prediction-card"
                if st.session_state.display_chart_mode == 'rating':
                    rating_card_class_single += " selected"
                
                st.markdown(
                    f"""
                    <div class="{rating_card_class_single}" onclick="
                        const button = document.querySelector('[data-testid="stButton"] button[id^="trigger_chart_rating_"]');
                        if (button) button.click();
                    ">
                        <h4>평점 분포</h4>
                        <p>{current_actual_rating:.1f} 점</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                st.button(
                    "평점 분포", 
                    key=f"trigger_chart_rating_{st.session_state.get('chart_rating_click_counter', 0)}", 
                    on_click=set_display_chart_mode, args=('rating',),
                    help="이 버튼은 숨겨져 있으며, 위 카드를 클릭하면 동작합니다.",
                    type="secondary"
                )
                st.session_state.chart_rating_click_counter = st.session_state.get('chart_rating_click_counter', 0) + 1

            st.markdown("""
                <style>
                    [data-testid="stButton"] button[id^="trigger_chart_audience_"],
                    [data-testid="stButton"] button[id^="trigger_chart_rating_"] {
                        display: none !important;
                    }
                </style>
            """, unsafe_allow_html=True)

            st.markdown("---")

            # --- 비교할 영화 추가 섹션 ---
            st.subheader("➕ 비교할 영화 추가")
            st.write("비교 그래프에 추가할 영화를 선택하세요. 최대 5개의 영화를 추가할 수 있습니다.")

            available_for_compare = [
                movie for movie in merged_test_df['영화명'].unique().tolist()
                if movie != st.session_state.selected_movie and 
                   movie not in [m['영화명'] for m in st.session_state.charted_movies]
            ]
            available_for_compare.sort()

            compare_movie_options = ["비교할 영화를 선택하세요..."] + available_for_compare

            selected_compare_movie_from_selectbox = st.selectbox(
                "비교할 영화 선택:",
                compare_movie_options,
                key="compare_movie_selector",
                index=0,
                on_change=lambda: setattr(st.session_state, 'selected_compare_movie', st.session_state.compare_movie_selector)
            )

            def add_movie_to_charted_list():
                if st.session_state.selected_compare_movie != "비교할 영화를 선택하세요...":
                    selected_compare_movie_data = merged_test_df[merged_test_df['영화명'] == st.session_state.selected_compare_movie]
                    
                    if not selected_compare_movie_data.empty:
                        movie_to_add = {
                            '영화명': st.session_state.selected_compare_movie,
                            '예측_누적관객수': selected_compare_movie_data.iloc[0]['예측_누적관객수'],
                            '평점': selected_compare_movie_data.iloc[0]['평점']
                        }
                        
                        existing_movie_names_in_chart = [m['영화명'] for m in st.session_state.charted_movies]
                        
                        if (st.session_state.selected_compare_movie != st.session_state.selected_movie) and \
                           (st.session_state.selected_compare_movie not in existing_movie_names_in_chart):
                            
                            if len(st.session_state.charted_movies) >= 5:
                                st.session_state.charted_movies.pop(0) # 가장 오래된 것 제거
                            
                            st.session_state.charted_movies.append(movie_to_add)
                            st.success(f"'{st.session_state.selected_compare_movie}'가 비교 목록에 추가되었습니다! ({len(st.session_state.charted_movies)}/5)")
                            st.session_state.display_chart_mode = 'audience' # 추가 시 관객수 그래프로 기본 전환
                            st.session_state.selected_compare_movie = "비교할 영화를 선택하세요..." # selectbox 초기화
                        else:
                            if st.session_state.selected_compare_movie == st.session_state.selected_movie:
                                st.info(f"'{st.session_state.selected_compare_movie}'는 현재 메인으로 선택된 영화입니다. 비교 목록에 중복 추가할 수 없습니다.")
                            else:
                                st.info(f"'{st.session_state.selected_compare_movie}'는 이미 비교 목록에 있습니다.")
                    else:
                        st.warning(f"선택된 비교 영화 '{st.session_state.selected_compare_movie}'의 데이터를 찾을 수 없습니다.")
                else:
                    st.warning("비교 목록에 추가할 영화를 선택해주세요.")

            col_add_compare_movie, col_clear_compare_chart = st.columns(2)

            with col_add_compare_movie:
                st.button(f"'{st.session_state.selected_compare_movie}' 비교 목록에 추가", 
                               key="add_compare_movie_btn",
                               on_click=add_movie_to_charted_list,
                               disabled=(st.session_state.selected_compare_movie == "비교할 영화를 선택하세요...") 
                               )

            with col_clear_compare_chart:
                if st.button("비교 목록 초기화", key="clear_compare_chart_btn"):
                    st.session_state.charted_movies = []
                    st.session_state.display_chart_mode = 'audience' # 초기화 시 관객수 그래프로 전환
                    st.success("비교 목록이 초기화되었습니다.")

            if st.session_state.charted_movies:
                st.markdown("---")
                st.markdown("#### 현재 비교 목록:")
                movie_names_in_chart = [m['영화명'] for m in st.session_state.charted_movies]
                st.write(f"- {', '.join(movie_names_in_chart)}")
                st.write(f"현재 비교 영화 수: {len(st.session_state.charted_movies)} / 5")
            else:
                st.info("비교할 영화를 '비교 목록에 추가' 버튼으로 추가해보세요! (메인 영화는 항상 그래프에 포함됩니다.)")

            st.markdown("---")

            # --- 그래프 그리기 로직 ---
            movies_to_chart = []
            chart_title_prefix = ""

            selected_main_movie_data = merged_test_df[merged_test_df['영화명'] == st.session_state.selected_movie]
            if not selected_main_movie_data.empty:
                main_movie_info = selected_main_movie_data.iloc[0].to_dict()
                movies_to_chart.append(main_movie_info)
                chart_title_prefix = f"'{st.session_state.selected_movie}'"

            # 차트 목록에 있는 영화들 중 메인 영화가 아닌 것만 추가
            for movie_in_charted_list in st.session_state.charted_movies:
                if movie_in_charted_list['영화명'] != st.session_state.selected_movie:
                    movies_to_chart.append(movie_in_charted_list)
            
            # 비교할 영화가 2개 이상일 때 타이틀 변경
            if len(movies_to_chart) > 1:
                chart_title_prefix = "영화 비교"
            
            if not movies_to_chart:
                st.info("그래프를 표시할 영화가 없습니다. 위에서 영화를 선택하거나 비교 목록에 추가해주세요.")
                fig, ax = plt.subplots(figsize=(10, 6))
                apply_chart_theme(fig, ax)
                ax.text(0.5, 0.5, "그래프 데이터 없음", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16, color='#FFD700')
                ax.axis('off')
                st.pyplot(fig)
            else:
                if st.session_state.display_chart_mode == 'audience':
                    st.markdown(f"#### {chart_title_prefix} 예상 누적 관객수")
                    labels_audience = []
                    sizes_audience = []
                    
                    # Top 50% 평균을 항상 포함 (단, 예측 관객수 데이터가 0인 경우 제외)
                    if audience_기준값_merged > 0:
                        labels_audience.append(f"Top 50% 평균")
                        sizes_audience.append(audience_기준값_merged)
                    
                    for movie in movies_to_chart:
                        if movie['예측_누적관객수'] > 0: # 0보다 큰 관객수만 포함
                            labels_audience.append(f"{movie['영화명']}")
                            sizes_audience.append(movie['예측_누적관객수']) 
                    
                    # 모든 값이 0이거나 데이터가 없는 경우 처리
                    if not sizes_audience or sum(sizes_audience) == 0:
                        st.info("선택된 영화들의 예상 누적 관객수가 없어 원형 그래프를 그릴 수 없습니다. 예측 값이 0인 영화는 제외됩니다.")
                        fig_audience, ax_audience = plt.subplots(figsize=(10, 10))
                        apply_chart_theme(fig_audience, ax_audience)
                        ax_audience.text(0.5, 0.5, "예측 관객수 데이터 없음", horizontalalignment='center', verticalalignment='center', transform=ax_audience.transAxes, fontsize=16, color='#FFD700')
                        ax_audience.axis('off')
                        st.pyplot(fig_audience)
                    else:
                        fig_audience, ax_audience = plt.subplots(figsize=(10, 10))
                        colors_audience = sns.color_palette("Spectral", len(labels_audience)).as_hex()
                        
                        def make_autopct_audience(values, labels_map_indices):
                            def my_autopct(pct):
                                total = sum(values)
                                if total == 0: return ''
                                actual_value = total * pct / 100
                                # 가장 가까운 실제 값의 인덱스 찾기
                                idx = (np.abs(np.array(values) - actual_value)).argmin()
                                label_text = labels_map_indices[idx]
                                
                                if "Top 50% 평균" in label_text:
                                    return f"{label_text}\n({int(values[idx]):,}명)"
                                else:
                                    # 영화명과 함께 실제 예측 관객수 표시
                                    return f"{label_text}\n({int(values[idx]):,}명)"
                            return my_autopct

                        # autopct를 위한 레이블과 인덱스 매핑 생성
                        labels_map_for_autopct = {i: label for i, label in enumerate(labels_audience)}

                        wedges_audience, texts_audience, autotexts_audience = ax_audience.pie(
                            sizes_audience, 
                            colors=colors_audience, 
                            autopct=make_autopct_audience(sizes_audience, labels_map_for_autopct), 
                            shadow=True, 
                            startangle=90,
                            textprops={'fontsize': 12, 'color': 'black'}
                        )
                        for autotext in autotexts_audience:
                            autotext.set_color('black')
                            autotext.set_fontsize(14)
                        for text_obj in texts_audience:
                            text_obj.set_text('') # pie 차트 자체의 라벨은 숨기고 autopct만 사용

                        ax_audience.legend(wedges_audience, labels_audience,
                                            title="영화 및 기준",
                                            loc="center left",
                                            bbox_to_anchor=(1, 0, 0.5, 1),
                                            fontsize=12,
                                            labelcolor='#f0f0f0',
                                            title_fontsize=14,
                                            facecolor='#3f4451',
                                            edgecolor='#FFD700'
                                            )

                        ax_audience.axis('equal') 
                        ax_audience.set_title(f"{chart_title_prefix} 예상 누적 관객수", fontsize=16, color='#f0f0f0')
                        apply_chart_theme(fig_audience, ax_audience) # 차트 테마 적용
                        st.pyplot(fig_audience)


                elif st.session_state.display_chart_mode == 'rating':
                    st.markdown(f"#### {chart_title_prefix} 평점 분포")

                    labels_rating = []
                    sizes_rating = []
                    
                    for movie in movies_to_chart:
                        if movie['평점'] > 0: # 0보다 큰 평점만 포함
                            labels_rating.append(f"{movie['영화명']}")
                            sizes_rating.append(movie['평점']) 

                    if not sizes_rating or sum(sizes_rating) == 0:
                        st.info("선택된 영화들의 평점이 없어 원형 그래프를 그릴 수 없습니다. 평점 값이 0인 영화는 제외됩니다.")
                        fig_rating, ax_rating = plt.subplots(figsize=(10, 10))
                        apply_chart_theme(fig_rating, ax_rating)
                        ax_rating.text(0.5, 0.5, "평점 데이터 없음", horizontalalignment='center', verticalalignment='center', transform=ax_rating.transAxes, fontsize=16, color='#FFD700')
                        ax_rating.axis('off')
                        st.pyplot(fig_rating)
                    else:
                        fig_rating, ax_rating = plt.subplots(figsize=(10, 10))
                        colors_rating_palette = sns.color_palette("viridis", len(labels_rating)).as_hex()
                        
                        movie_colors_rating = {}
                        for i, movie_name_label in enumerate(labels_rating):
                            movie_colors_rating[movie_name_label] = colors_rating_palette[i % len(colors_rating_palette)]
                        
                        actual_colors_rating = [movie_colors_rating[label] for label in labels_rating]

                        def make_autopct_rating(values, labels_map_indices):
                            def my_autopct(pct):
                                total = sum(values)
                                if total == 0: return ''
                                actual_value_from_pct = total * pct / 100
                                idx = (np.abs(np.array(values) - actual_value_from_pct)).argmin()
                                
                                # 영화명과 함께 실제 평점 표시
                                return f'{labels_map_indices[idx]}\n({values[idx]:.1f}점)'
                            return my_autopct
                        
                        labels_map_for_autopct_rating = {i: label for i, label in enumerate(labels_rating)}

                        wedges_rating, texts_rating, autotexts_rating = ax_rating.pie(
                            sizes_rating, 
                            colors=actual_colors_rating, 
                            autopct=make_autopct_rating(sizes_rating, labels_map_for_autopct_rating),
                            shadow=True, 
                            startangle=90, 
                            textprops={'fontsize': 12, 'color': 'black'},
                            wedgeprops={'linewidth': 1, 'edgecolor': '#2c313d'}
                        )
                        
                        for autotext in autotexts_rating:
                            autotext.set_color('black')
                            autotext.set_fontsize(14) 
                        for text_obj in texts_rating:
                            text_obj.set_text('')

                        ax_rating.legend(wedges_rating, labels_rating,
                                            title="영화 평점",
                                            loc="center left",
                                            bbox_to_anchor=(1, 0, 0.5, 1),
                                            fontsize=12,
                                            labelcolor='#f0f0f0',
                                            title_fontsize=14,
                                            facecolor='#3f4451',
                                            edgecolor='#FFD700'
                                            )

                        ax_rating.axis('equal') 
                        ax_rating.set_title(f"{chart_title_prefix} 평점 비교", fontsize=16, color='#f0f0f0') 
                        apply_chart_theme(fig_rating, ax_rating) # 차트 테마 적용
                        st.pyplot(fig_rating)
                        
        else:
            st.warning(f"선택하신 영화 '{st.session_state.selected_movie}'에 대한 예측 결과 데이터(예측 누적 관객수 또는 평점)를 'merged_test.csv' 파일에서 찾을 수 없습니다.")
    else:
        st.info("위에서 영화를 선택하여 상세 정보, 추천 영화 및 예측 결과를 확인하세요.")