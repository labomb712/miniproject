import pandas as pd
import os
import streamlit as st

@st.cache_data(show_spinner="🎞️ 영화 데이터를 불러오는 중입니다...")
def load_data(file_path):
    """
    CSV 파일에서 영화 데이터를 로드하고 기본 전처리를 수행합니다.
    """
    if not os.path.exists(file_path):
        st.error(f"오류: 데이터 파일 '{file_path}'을(를) 찾을 수 없습니다. 'data' 폴더에 파일을 넣어주세요.")
        st.stop() # 필수 파일이 없으면 앱 중단

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

@st.cache_data(show_spinner="⏳ 예측 결과 데이터를 불러오는 중입니다...")
def load_and_preprocess_merged_test_data(file_path):
    """
    merged_test.csv 파일을 로드하고 예측 결과 표시를 위한 전처리를 수행합니다.
    """
    if not os.path.exists(file_path):
        st.error(f"오류: 데이터 파일 '{file_path}'을(를) 찾을 수 없습니다. 'data' 폴더에 'merged_test.csv' 파일을 넣어주세요.")
        return pd.DataFrame() 

    merged_df = pd.read_csv(file_path)
    
    required_cols = ['영화명', '예측_누적관객수', '평점']
    if not all(col in merged_df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in merged_df.columns]
        st.error(f"파일에 필요한 다음 컬럼이 없습니다: {', '.join(missing_cols)}. 파일을 확인해주세요.")
        return pd.DataFrame()

    for col in ['누적관객수', '누적매출액', '예측_누적관객수', '평점']:
        if col in merged_df.columns:
            merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').fillna(0)
    
    if '개봉일' in merged_df.columns:
        merged_df['개봉일'] = pd.to_datetime(merged_df['개봉일'], errors='coerce', format='%Y-%m-%d')
    else:
        st.warning("'개봉일' 컬럼이 없어 날짜 기반 필터링이나 정보 표시가 제한될 수 있습니다.")
    
    merged_df.dropna(subset=['영화명'], inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    
    return merged_df