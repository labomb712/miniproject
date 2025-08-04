import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import platform
import os
import streamlit as st

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

def apply_chart_theme(fig, ax):
    """
    Matplotlib 차트에 Streamlit 앱 테마를 적용합니다.
    """
    fig.patch.set_facecolor('#F5F7FA') # 앱 배경색과 동일
    ax.set_facecolor('#F5F7FA') # 플롯 영역 배경색과 동일
    
    # 축 라벨, 눈금, 스파인 색상 설정
    ax.tick_params(axis='x', colors='#f0f0f0')
    ax.tick_params(axis='y', colors='#f0f0f0')
    ax.spines['left'].set_color('#f0f0f0')
    ax.spines['bottom'].set_color('#f0f0f0')
    ax.spines['right'].set_color('#f0f0f0')
    ax.spines['top'].set_color('#f0f0f0')

    # 라벨 및 타이틀 색상 설정
    if ax.xaxis.get_label_text():
        ax.set_xlabel(ax.xaxis.get_label_text(), color='#1F2937')
    if ax.yaxis.get_label_text():
        ax.set_ylabel(ax.yaxis.get_label_text(), color='#1F2937')
    if ax.get_title():
        ax.set_title(ax.get_title(), color='#1F2937')

    # 범례 텍스트 색상 및 배경 색상 설정
    if ax.get_legend():
        for text in ax.get_legend().get_texts():
            text.set_color('#1F2937')
        ax.get_legend().get_frame().set_facecolor('#f0f0f0') # 범례 배경색
        ax.get_legend().get_frame().set_edgecolor('#FFD700') # 범례 테두리색
    
    # 그리드 색상 설정
    ax.grid(True, color='#5a5f6e', linestyle=':', alpha=0.7)