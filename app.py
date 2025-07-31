import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🎬 영화제목")

# 데이터 로드
df = pd.read_csv('data/kobis_boxoffice_latest_202101_202507.csv')

# 간단 통계
total = len(df)
positive = len(df[df['predicted'] == 1])
negative = total - positive
positive_ratio = round(positive / total * 100, 2)

# 시각화
st.subheader("")
st.write(f"")
st.write(f"")
st.write(f"")

fig, ax = plt.subplots()
sns.countplot(x='predicted', data=df, ax=ax)
ax.set_xticklabels([])
st.pyplot(fig)


st.subheader("")
st.dataframe()