import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os
import matplotlib.pyplot as plt
import seaborn as sns
import platform

# -----------------------------------------------------------------------------
# 1. 페이지 기본 설정
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="통영 적조 예측 시스템",
    page_icon="🌊",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2. 한글 폰트 설정
# -----------------------------------------------------------------------------
system_name = platform.system()
if system_name == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif system_name == 'Darwin': # Mac
    plt.rc('font', family='AppleGothic')
else: # Linux (Streamlit Cloud)
    try:
        plt.rc('font', family='NanumGothic')
    except:
        pass
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------------------------------------------------------
# 3. 데이터 로드 함수
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    possible_paths = [
        "tongyeong_lite.csv",
        "/content/tongyeong_lite.csv",
        "/content/drive/MyDrive/redtide_project/tongyeong_lite.csv"
    ]
    
    file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            break
    
    if file_path is None:
        return None
    
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        # 이상치 제거
        df = df[(df['Temp'] > 0) & (df['Salt'] > 0) & (df['Salt'] < 45)]
        return df
    except Exception as e:
        st.error(f"데이터 오류: {e}")
        return None

# -----------------------------------------------------------------------------
# 4. 적조 위험도 진단 로직 (사용자 수정 반영됨)
# -----------------------------------------------------------------------------
def assess_red_tide_risk(temp, salt):
    risk_score = 0
    reasons = []

    # 수온 평가
    if 25 <= temp <= 28:
        risk_score += 50
        reasons.append("🌡️ **수온(25~28℃)**: 적조 생물 증식에 최적입니다.")
    elif 28 <= temp <= 30:
        risk_score += 30
        reasons.append("🌡️ **고수온(28~30℃)**: 성장이 다소 둔화될 수 있으나 주의가 필요합니다.")
    elif temp >= 30:
        risk_score += 30
        reasons.append("🌡️ **과고수온(30℃↑)**: 적조 생물 성장이 확연히 저하됩니다.")
    elif 19 <= temp <= 25:
        risk_score += 30
        reasons.append("❄️ **저수온(19~25℃)**: 수온이 낮아 적조 발생 확률이 낮습니다.")
    elif temp <= 19:
        risk_score += 30
        reasons.append("❄️ **과저수온(19℃↓)**: 적조 생물 성장이 확연히 저하됩니다.")

    # 염분 평가
    if 33 <= salt <= 35:
        risk_score += 50
        reasons.append("🧂 **염분(33~35psu)**: 적조 생물 증식에 최적입니다.")
    elif salt <= 32:
        risk_score -= 20
        reasons.append("💧 **저염분(32psu↓)**: 염분이 너무 낮아 적조 생물의 성장이 특히 저하됩니다.")
    else:
        reasons.append("🧂 **염분**: 적조 발생 최적 범위를 벗어났습니다.")

    # 최종 진단
    if risk_score >= 90:
        return "🚨 매우 위험 (적조 대발생 가능)", "red", reasons
    elif risk_score >= 50:
        return "⚠️ 주의 (적조 발생 가능 조건 충족)", "orange", reasons
    else:
        return "✅ 안전 (적조 발생 확률 낮음)", "green", reasons

# -----------------------------------------------------------------------------
# 5. 메인 화면 구성
# -----------------------------------------------------------------------------
def main():
    st.title("🌊 통영 적조 예측 및 분석 시스템")
    st.markdown("##### 지난 23년간(2001-2023)의 통영 조위관측소 빅데이터 기반")
    
    with st.sidebar:
        st.header("데이터 현황")
        with st.spinner("데이터 로딩 중..."):
            df = load_data()
        
        if df is not None:
            st.success("연결 성공!")
            st.metric("총 데이터", f"{len(df):,} 건")
            st.metric("분석 기간", f"{df.index.min().year} ~ {df.index.max().year}")
        else:
            st.error("데이터 없음")
            st.warning("tongyeong_lite.csv 파일을 찾을 수 없습니다.")
            st.stop()

    tab1, tab2, tab3 = st.tabs(["📅 과거 날짜 조회", "🔮 수온 기반 예측", "📊 데이터 분포"])

    # [탭 1] 과거 조회
    with tab1:
        st.subheader("과거 바다 상태 조회")
        col1, col2 = st.columns([1, 2])
        with col1:
            min_d = df.index.min().date()
            max_d = df.index.max().date()
            default_d = pd.to_datetime(f"{max_d.year-1}-08-15").date()
            input_date = st.date_input("날짜 선택", value=default_d, min_value=min_d, max_value=max_d)
            if st.button("조회하기", type="primary", key='btn1', use_container_width=True):
                target_data = df[df.index.date == input_date]
                if len(target_data) > 0:
                    avg_t = target_data['Temp'].mean()
                    avg_s = target_data['Salt'].mean()
                    level, color, reasons = assess_red_tide_risk(avg_t, avg_s)
                    
                    with col2:
                        st.markdown(f"### {input_date} 분석 결과")
                        m1, m2 = st.columns(2)
                        m1.metric("평균 수온", f"{avg_t:.2f} ℃")
                        m2.metric("평균 염분", f"{avg_s:.2f} psu")
                        st.markdown(f"#### 진단: :{color}[{level}]")
                        with st.expander("상세 진단 근거 보기", expanded=True):
                            for r in reasons: st.write(f"- {r}")
                else:
                    with col2: st.warning("해당 날짜의 데이터가 없습니다.")

    # [탭 2] 수온 예측
    with tab2:
        st.subheader("수온 기반 적조 예측")
        col_in, col_out = st.columns([1, 2])
        with col_in:
            input_temp = st.number_input("수온 입력 (℃)", value=25.5, step=0.1)
            if st.button("예측 실행", type="primary", key='btn2', use_container_width=True):
                X = df[['Temp']]
                y = df['Salt']
                model = LinearRegression()
                model.fit(X, y)
                pred_salt = model.predict([[input_temp]])[0]
                level, color, reasons = assess_red_tide_risk(input_temp, pred_salt)
                
                with col_out:
                    st.markdown(f"### 예측 결과 (수온 {input_temp}℃ 기준)")
                    st.metric("예상 염분", f"{pred_salt:.2f} psu")
                    st.markdown(f"#### 진단: :{color}[{level}]")
                    st.info("💡 **분석 근거:**\n\n" + "\n".join([f"- {r}" for r in reasons]))

    # [탭 3] 시각화
    with tab3:
        st.subheader("통영 해역 수온-염분 분포")
        if st.checkbox("산점도 그래프 보기", value=True):
            fig, ax = plt.subplots(figsize=(10, 6))
            sample = df.sample(min(len(df), 5000))
            
            # x, y에는 데이터의 실제 영어 컬럼명('Temp', 'Salt')을 넣어야 합니다.
            sns.scatterplot(data=sample, x='Temp', y='Salt', alpha=0.15, color='teal', ax=ax, s=15, label='관측 데이터')
            
            # 축 제목을 여기서 한글로 바꿔줍니다.
            ax.set_xlabel("수온 (℃)")
            ax.set_ylabel("염분 (psu)")
            
            import matplotlib.patches as patches
            # 사용자 로직에 맞춰 박스 구간 수정 (25~28도, 33~35psu)
            rect = patches.Rectangle((25, 33), 3, 2, linewidth=2, edgecolor='red', facecolor='none', label='적조 최적 구간')
            ax.add_patch(rect)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

if __name__ == "__main__":
    main()
