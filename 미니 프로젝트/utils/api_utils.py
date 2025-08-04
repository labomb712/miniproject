import requests
import streamlit as st
import os

@st.cache_data(show_spinner=False)
def get_movie_poster_url(movie_title):
    """
    TMDB API를 사용하여 영화 포스터 URL을 가져옵니다.
    """
    # TMDB API 키는 보안상 Streamlit Secrets 또는 환경 변수로 관리하는 것이 좋습니다.
    # st.secrets["tmdb_api_key"] 또는 os.environ.get("TMDB_API_KEY") 사용 권장
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
        # 오류 메시지를 사용자에게 표시하지 않고 내부적으로만 처리 (예: 로깅)
        pass 
    return "https://placehold.co/300x450/cccccc/000000?text=No+Image"