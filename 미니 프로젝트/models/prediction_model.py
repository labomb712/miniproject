import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# plot_utils에서 차트 테마 함수 임포트
from utils.plot_utils import apply_chart_theme

def train_and_predict_xgboost_model(df_data):
    """
    XGBoost 모델을 학습하고 예측 성능을 시각화하여 반환합니다.
    Args:
        df_data (pd.DataFrame): 원본 영화 데이터프레임.
    Returns:
        plt.Figure: 예측 결과 산점도 그래프.
    """
    xgb_features = ['감독', '제작국가', '장르', '개봉년도', '개봉월', '개봉요일', '누적매출액']
    xgb_target = '누적관객수'

    required_for_xgb = xgb_features + [xgb_target]
    if not all(col in df_data.columns for col in required_for_xgb):
        missing_cols = [col for col in required_for_xgb if col not in df_data.columns]
        st.error(f"XGBoost 모델 학습에 필요한 다음 컬럼이 없습니다: {', '.join(missing_cols)}. 데이터 파일을 확인해주세요.")
        fig, ax = plt.subplots(figsize=(10, 6))
        apply_chart_theme(fig, ax) # 테마 적용
        ax.text(0.5, 0.5, "필수 데이터 컬럼 누락", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16, color='#FFD700')
        ax.axis('off')
        return fig

    xgb_df = df_data.copy()

    le = LabelEncoder() 
    for col in ['감독', '제작국가', '장르']:
        # 이미 변환된 데이터가 있을 수 있으므로 오류 방지
        try:
            xgb_df[col] = le.fit_transform(xgb_df[col].astype(str)) 
        except Exception as e:
            st.warning(f"LabelEncoder 변환 중 오류 발생 ({col}): {e}. 해당 컬럼을 스킵합니다.")
            xgb_df[col] = 0 # 임시로 0으로 채우거나 다른 방법 강구

    X_xgb = xgb_df[xgb_features]
    y_xgb = xgb_df[xgb_target]

    if X_xgb.empty or len(X_xgb) < 2:
        st.warning("XGBoost 모델 학습을 위한 데이터가 충분하지 않습니다. 파일 내용과 전처리 결과를 확인해주세요.")
        fig, ax = plt.subplots(figsize=(10, 6))
        apply_chart_theme(fig, ax)
        ax.text(0.5, 0.5, "데이터 부족으로 예측 불가", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16, color='#FFD700')
        ax.axis('off')
        return fig
    else:
        try:
            X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
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

            st.subheader("📊 모델 성능 지표 (학습 데이터의 테스트 세트)")
            col1, col2, col3, col4 = st.columns(4) 
            col1.metric("MSE", f"{mean_squared_error(y_test_xgb, y_pred_xgb):,.0f}")
            col2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test_xgb, y_pred_xgb)):,.0f}")
            col3.metric("MAE", f"{mean_absolute_error(y_test_xgb, y_pred_xgb):,.0f}") 
            col4.metric("R² Score", f"{r2_score(y_test_xgb, y_pred_xgb):.4f}")

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=y_test_xgb, y=y_pred_xgb, alpha=0.6, ax=ax, color='#66b2ff')
            ax.plot([y_test_xgb.min(), y_test_xgb.max()], [y_test_xgb.min(), y_test_xgb.max()], 'r--', lw=2, label='이상적인 예측')
            ax.set_xlabel("실제 누적 관객수", color='#f0f0f0')
            ax.set_ylabel("예측 누적 관객수", color='#f0f0f0')
            ax.legend(labelcolor='#f0f0f0')
            
            apply_chart_theme(fig, ax) # 테마 적용

            ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
            ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
            plt.xticks(rotation=45)
            return fig
        except Exception as e: 
            st.error(f"XGBoost 모델 학습 또는 예측 중 오류 발생: {e}. 데이터셋 크기 또는 특성을 확인해주세요.")
            fig, ax = plt.subplots(figsize=(10, 6))
            apply_chart_theme(fig, ax)
            ax.text(0.5, 0.5, "모델 학습 중 오류 발생", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16, color='#FFD700')
            ax.axis('off')
            return fig

def calculate_audience_benchmark(merged_df):
    """
    merged_test_df의 상위 50% 관객수 영화들의 평균을 계산합니다.
    """
    audience_values_merged = merged_df['예측_누적관객수'][merged_df['예측_누적관객수'] > 0].sort_values(ascending=False)
    if not audience_values_merged.empty:
        top_50_percent_index_merged = int(len(audience_values_merged) * 0.5)
        return audience_values_merged.iloc[:top_50_percent_index_merged].mean()
    else:
        return 1000000 # 기본값 (데이터 없을 경우)