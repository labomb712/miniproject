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
    rc('font', family='AppleGothic')
else:
    rc('font', family='DejaVu Sans')
plt.rcParams['axes.unicode_minus'] = False

# Streamlit 앱 시작
st.set_page_config(page_title="영화 예측 시스템", layout="wide")
st.title("영화 예측 시스템")

# 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_csv("data/kobis_boxoffice_latest_202101_202507.csv")
    df['audiAcc'] = pd.to_numeric(df['audiAcc'], errors='coerce')
    df['salesAcc'] = pd.to_numeric(df['salesAcc'], errors='coerce')
    df['targetDt'] = pd.to_datetime(df['targetDt'], errors='coerce', format='%Y%m%d')
    df.dropna(subset=['audiAcc', 'salesAcc', 'targetDt'], inplace=True)
    df['개봉_월'] = df['targetDt'].dt.month
    return df

df = load_data()

# 영화 검색 기능
st.subheader("영화 검색")
search_input = st.text_input("검색할 영화 제목을 입력하세요")

if search_input:
    result_df = df[df['movieNm'].str.contains(search_input, case=False, na=False)]
    if not result_df.empty:
        st.success(f"{len(result_df)}개의 검색 결과가 있습니다:")
        st.dataframe(result_df[['movieNm', 'audiAcc', 'salesAcc', 'targetDt']])
    else:
        st.warning("검색 결과가 없습니다.")

# 피처 및 타겟 설정
X = df[['salesAcc', '개봉_월']]
y = df['audiAcc']

numerical_features = ['salesAcc']
categorical_features = ['개봉_월']

# 전처리
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# 모델
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

# 데이터 분할
test_size_val = max(0.2, 1 / len(X))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_val, random_state=42)

# 학습
model_pipeline.fit(X_train, y_train)

# 예측
y_pred = model_pipeline.predict(X_test)

# 평가
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# 평가 지표 출력
st.subheader("모델 성능 지표")
st.markdown(f"- **MSE:** {mse:,.2f}")
st.markdown(f"- **RMSE:** {rmse:,.2f}")
st.markdown(f"- **R² Score:** {r2:.4f}")

# 예측 시각화
st.subheader("실제 vs 예측 관객수 시각화")

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, ax=ax)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel("실제 누적 관객수")
ax.set_ylabel("예측 누적 관객수")
ax.set_title("랜덤 포레스트 회귀: 실제 vs 예측")
ax.grid(True)
st.pyplot(fig)