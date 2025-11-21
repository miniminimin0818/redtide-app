import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os
import matplotlib.pyplot as plt
import seaborn as sns
import platform

# -----------------------------------------------------------------------------
# 1. 페이지 기본 설정 (반드시 코드 최상단에 위치해야 함)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="통영 적조 예측 시스템",
    page_icon="🌊",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2. 한글 폰트 설정 (운영체제별 자동 대응)
# -----------------------------------------------------------------------------
system_name = platform.system()
if system_name == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif system_name == 'Darwin': # Mac
    plt.rc('font', family='AppleGothic')
else: # Linux (Colab, Streamlit Cloud)
    # 나눔폰트가 설치되어 있다면 사용, 아니면 기본 폰트 유지
    try:
        plt.rc('font', family='NanumGothic')
    except:
        pass
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------------------------------------------------------
# 3. 데이터 로드 함수 (캐싱 적용으로 속도 최적화)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    """
    tongyeong_lite.csv 파일을 찾아 읽어옵니다.
    로컬, 코랩, 구글 드라이브 경로를 순차적으로 확인합니다.
    """
    # 파일이 있을 법한 경로들을 순서대로 확인
    possible_paths = [
        "tongyeong_lite.csv",                                         # 1. 현재 폴더 (GitHub/로컬)
        "/content/tongyeong_lite.csv",                                # 2. 구글 코랩 최상위
        "/content/drive/MyDrive/redtide_project/tongyeong_lite.csv"   # 3. 구글 드라이브
    ]
    
    file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            break
    
    if file_path is None:
        return None
    
    try:
        # CSV 읽기
        df = pd.read_csv(file_path)
        # 날짜 컬럼 변환 및 인덱스 설정
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # 데이터 정제: 수온이나 염분이 0 이하인 이상치 제거
        df = df[(df['Temp'] > 0) & (df['Salt'] > 0) & (df['Salt'] < 45)]
        return df
    except Exception as e:
        st.error(f"데이터를 읽는 중 오류가 발생했습니다: {e}")
        return None

# -----------------------------------------------------------------------------
# 4. 적조 위험도 진단 로직 (핵심 알고리즘)
# -----------------------------------------------------------------------------
def assess_red_tide_risk(temp, salt):
    """
    수온(temp)과 염분(salt)을 입력받아 적조(코클로디니움) 발생 위험도를 진단합니다.
    """
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

    # 최종 진단 등급 산정
    if risk_score >= 90:
        return "🚨 매우 위험 (적조 대발생 가능)", "red", reasons
    elif risk_score >= 50:
        return "⚠️ 주의 (적조 발생 가능 조건 충족)", "orange", reasons
    else:
        return "✅ 안전 (적조 발생 확률 낮음)", "green", reasons

# -----------------------------------------------------------------------------
# 5. 메인 화면 구성 (UI)
# -----------------------------------------------------------------------------
def main():
    st.title("🌊 통영 적조 예측 및 분석 시스템")
    st.markdown("##### 지난 23년간(2001-2023)의 통영 조위관측소 빅데이터 기반")
    
    # --- 사이드바: 데이터 로딩 상태 ---
    with st.sidebar:
        st.header("데이터 현황")
        with st.spinner("데이터 로딩 중..."):
            df = load_data()
        
        if df is not None:
            st.success("연결 성공!")
            st.metric("총 데이터", f"{len(df):,} 건")
            st.metric("분석 기간", f"{df.index.min().year} ~ {df.index.max().year}")
            st.info("현재 'tongyeong_lite.csv' 데이터를 사용 중입니다.")
        else:
            st.error("데이터 없음")
            st.warning("`tongyeong_lite.csv` 파일을 찾을 수 없습니다.")
            st.markdown("""
            **[해결 방법]**
            1. `make_lite_data.py`를 실행해 CSV 파일을 만드셨나요?
            2. GitHub에 올리셨다면 `tongyeong_lite.csv`도 같이 올리셨나요?
            3. 코랩이라면 왼쪽 파일 목록에 CSV 파일을 업로드하셨나요?
            """)
            st.stop() # 데이터 없으면 여기서 중단

    # --- 메인 탭 메뉴 구성 ---
    tab1, tab2, tab3 = st.tabs(["📅 과거 날짜 조회", "🔮 수온 기반 예측", "📊 데이터 분포"])

    # [탭 1] 과거 날짜 조회 기능
    with tab1:
        st.subheader("과거 바다 상태 조회")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            min_d = df.index.min().date()
            max_d = df.index.max().date()
            # 기본값: 데이터 마지막 연도의 8월 15일 (적조 빈번 시기)
            default_d = pd.to_datetime(f"{max_d.year-1}-08-15").date()
            
            input_date = st.date_input("날짜 선택", value=default_d, min_value=min_d, max_value=max_d)
            # use_container_width=True로 버튼을 꽉 차게 만듦
            if st.button("조회하기", type="primary", key='btn1', use_container_width=True):
                # 해당 날짜 데이터 필터링
                target_data = df[df.index.date == input_date]
                
                if len(target_data) > 0:
                    avg_t = target_data['Temp'].mean()
                    avg_s = target_data['Salt'].mean()
                    level, color, reasons = assess_red_tide_risk(avg_t, avg_s)
                    
                    # 결과 화면 갱신 (오른쪽 컬럼)
                    with col2:
                        st.markdown(f"### {input_date.strftime('%Y년 %m월 %d일')} 분석 결과")
                        
                        m1, m2 = st.columns(2)
                        m1.metric("평균 수온", f"{avg_t:.2f} ℃")
                        m2.metric("평균 염분", f"{avg_s:.2f} psu")
                        
                        st.markdown(f"#### 진단: :{color}[{level}]")
                        with st.expander("상세 진단 근거 보기", expanded=True):
                            for r in reasons:
                                st.write(f"- {r}")
                else:
                    with col2:
                        st.warning("해당 날짜의 관측 데이터가 없습니다.")

    # [탭 2] 수온 기반 예측 기능 (회귀분석)
    with tab2:
        st.subheader("수온 기반 적조 예측")
        st.caption("현재 수온을 입력하면, 과거 통계 데이터를 바탕으로 염분을 예측하고 적조 위험을 알려줍니다.")
        
        col_in, col_out = st.columns([1, 2])
        with col_in:
            input_temp = st.number_input("수온 입력 (℃)", value=25.5, step=0.1, min_value=0.0, max_value=35.0)
            if st.button("예측 실행", type="primary", key='btn2', use_container_width=True):
                # 선형 회귀 모델 학습 (전체 데이터 사용)
                X = df[['Temp']]
                y = df['Salt']
                model = LinearRegression()
                model.fit(X, y)
                
                # 예측 수행
                pred_salt = model.predict([[input_temp]])[0]
                level, color, reasons = assess_red_tide_risk(input_temp, pred_salt)
                
                # 결과 화면 갱신 (오른쪽 컬럼)
                with col_out:
                    st.markdown(f"### 예측 결과 (수온 {input_temp}℃ 기준)")
                    st.metric("예상 염분", f"{pred_salt:.2f} psu")
                    
                    st.markdown(f"#### 진단: :{color}[{level}]")
                    st.info("💡 **분석 근거:**\n\n" + "\n".join([f"- {r}" for r in reasons]))
                    st.caption("* 이 결과는 지난 23년 데이터의 수온-염분 상관관계를 기반으로 추산되었습니다.")

    # [탭 3] 데이터 시각화 기능
    with tab3:
        st.subheader("통영 해역 수온-염분 분포")
        st.caption("23년간의 전체 데이터를 시각화하여 보여줍니다.")
        
        if st.checkbox("산점도 그래프 보기", value=True):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 데이터가 너무 많으면 느려지므로 5000개만 랜덤 샘플링해서 그리기
            sample = df.sample(min(len(df), 5000))
            
            # 산점도 그리기
            sns.scatterplot(data=sample, x="온도", y="염분", alpha=0.15, color='teal', ax=ax, s=15, label="관측 데이터")
            
            # 적조 위험 구간 (빨간 네모 박스) 표시
            import matplotlib.patches as patches
            # (x시작, y시작), 너비, 높이 -> 수온 24~27도, 염분 30~34psu
            rect = patches.Rectangle((24, 30), 3, 4, linewidth=2, edgecolor='red', facecolor='none', label="적조 위험 구간")
            ax.add_patch(rect)
            
            ax.set_xlabel("수온 (℃)")
            ax.set_ylabel("염분 (psu)")
            ax.set_title(f"수온 vs 염분 상관관계 (샘플 {len(sample)}개)")
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)

# 프로그램 시작점
if __name__ == "__main__":

    main()
