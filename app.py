import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 한글 폰트 설정
import platform
from matplotlib import font_manager, rc

if platform.system() == 'Windows':
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)
elif platform.system() == 'Darwin':
    # Mac 운영체제인 경우 AppleGothic 폰트 설정
    rc('font', family='AppleGothic')
else:
    # 그 외 운영체제인 경우 DejaVu Sans 폰트 설정 (일반적으로 리눅스 환경)
    rc('font', family='DejaVu Sans')
# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# Streamlit 앱 설정
st.set_page_config(page_title="영화 예측 시스템", layout="wide")
st.title("영화 예측 시스템")

# 데이터 불러오기 함수
@st.cache_data # Streamlit 캐싱 데코레이터: 데이터를 한 번 로드하면 다시 로드하지 않음
def load_data():
    # CSV 파일 로드 (파일 이름이 한글이므로 정확히 일치해야 함)
    df = pd.read_csv("20-25년_영화데이터_한글컬럼.csv")

    # '누적관객수' 컬럼을 숫자형으로 변환 (변환 불가 시 NaN으로 처리)
    df['누적관객수'] = pd.to_numeric(df['누적관객수'], errors='coerce')
    # '누적매출액' 컬럼을 숫자형으로 변환 (변환 불가 시 NaN으로 처리)
    df['누적매출액'] = pd.to_numeric(df['누적매출액'], errors='coerce')
    # '개봉일' 컬럼을 날짜/시간 형식으로 변환 (변환 불가 시 NaN으로 처리, YYYYMMDD 형식 지정)
    df['개봉일'] = pd.to_datetime(df['개봉일'], errors='coerce', format='%Y-%m-%d') # CSV snippet suggests YYYY-MM-DD format

    # 예측에 필요한 핵심 컬럼에 NaN 값이 있는 행 제거
    df.dropna(subset=['누적관객수', '누적매출액', '개봉일'], inplace=True)

    # '개봉일' 컬럼에서 월(month) 정보 추출하여 '개봉_월' 컬럼 생성
    df['개봉_월'] = df['개봉일'].dt.month
    return df

# 데이터 로드
df = load_data()

# 영화 검색 기능 섹션
st.subheader("영화 검색")
search_input = st.text_input("검색할 영화 제목을 입력하세요")

if search_input:
    # '영화명' 컬럼에서 검색어 포함 여부 확인 (대소문자 구분 없이, NaN 값은 무시)
    result_df = df[df['영화명'].str.contains(search_input, case=False, na=False)]
    if not result_df.empty:
        # 검색 결과가 있을 경우 결과 개수와 데이터프레임 표시
        st.success(f"{len(result_df)}개의 검색 결과가 있습니다:")
        st.dataframe(result_df[['영화명', '누적관객수', '누적매출액', '개봉일']])
    else:
        # 검색 결과가 없을 경우 경고 메시지 표시
        st.warning("검색 결과가 없습니다.")

# 피처(독립 변수) 및 타겟(종속 변수) 설정
X = df[['누적매출액', '개봉_월']] # 예측에 사용할 피처: 누적매출액, 개봉_월
y = df['누적관객수'] # 예측할 타겟: 누적관객수

# 숫자형 피처와 범주형 피처 정의
numerical_features = ['누적매출액']
categorical_features = ['개봉_월']

# 데이터 전처리 파이프라인 설정
preprocessor = ColumnTransformer(
    transformers=[
        # 숫자형 피처에 StandardScaler 적용 (평균 0, 분산 1로 스케일링)
        ('num', StandardScaler(), numerical_features),
        # 범주형 피처에 OneHotEncoder 적용 (원-핫 인코딩, 학습 시 보지 못한 범주는 무시)
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# 모델 파이프라인 설정 (전처리기 + 회귀 모델)
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor), # 전처리 단계
    # RandomForestRegressor 사용 (100개의 트리, 재현성을 위한 random_state, 모든 코어 사용)
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

# 데이터 분할
# 전체 데이터셋의 20%를 테스트셋으로 사용하거나, 데이터가 적으면 최소 1개라도 테스트셋에 포함
test_size_val = max(0.2, 1 / len(X))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_val, random_state=42)

# 모델 학습
model_pipeline.fit(X_train, y_train)

# 테스트 데이터에 대한 예측 수행
y_pred = model_pipeline.predict(X_test)

# 모델 성능 평가 지표 계산
mse = mean_squared_error(y_test, y_pred) # 평균 제곱 오차 (Mean Squared Error)
rmse = np.sqrt(mse) # 제곱근 평균 제곱 오차 (Root Mean Squared Error)
r2 = r2_score(y_test, y_pred) # 결정 계수 (R-squared Score)

# 평가 지표 출력 섹션
st.subheader("모델 성능 지표")
st.markdown(f"- **MSE (평균 제곱 오차):** {mse:,.2f}")
st.markdown(f"- **RMSE (제곱근 평균 제곱 오차):** {rmse:,.2f}")
st.markdown(f"- **R² Score (결정 계수):** {r2:.4f}")

# 예측 결과 시각화 섹션
st.subheader("실제 vs 예측 관객수 시각화")

# Matplotlib figure와 axes 생성
fig, ax = plt.subplots(figsize=(10, 6))
# 실제 관객수와 예측 관객수를 산점도로 표시
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, ax=ax)
# 완벽한 예측을 나타내는 대각선 (y=x) 추가
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
# 축 레이블 설정
ax.set_xlabel("실제 누적 관객수")
ax.set_ylabel("예측 누적 관객수")
# 그래프 제목 설정
ax.set_title("랜덤 포레스트 회귀: 실제 vs 예측")
# 그리드 추가
ax.grid(True)
# Streamlit에 그래프 표시
st.pyplot(fig)