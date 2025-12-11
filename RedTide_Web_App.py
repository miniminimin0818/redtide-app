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
else: # Linux (Colab, Streamlit Cloud)
    try:
        plt.rc('font', family='NanumGothic')
    except:
        pass
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------------------------------------------------------
# 3. 데이터 로드 함수
# -----------------------------------------------------------------------------
@st.cache_data
def load_all_data():
    paths = [
        ".", 
        "/content",
    ]
    
    env_df = None   # 일반 환경 데이터 (tongyeong_lite.csv)
    occur_df = None # 적조 발생 데이터 (redtide_occurrences.csv)
    
    # 1. 전체 환경 데이터 로드
    for p in paths:
        fpath = os.path.join(p, "tongyeong_lite.csv")
        if os.path.exists(fpath):
            try:
                env_df = pd.read_csv(fpath)
                env_df['Date'] = pd.to_datetime(env_df['Date'])
                env_df.set_index('Date', inplace=True)
                # 이상치 제거
                env_df = env_df[(env_df['Temp'] > 0) & (env_df['Salt'] > 0) & (env_df['Salt'] < 45)]
                # 월-일 정보 추가 (미래 예측용)
                env_df['MM-DD'] = env_df.index.strftime('%m-%d')
                break
            except: pass
            
    # 2. 적조 발생 데이터 로드 (밀도 정보 포함)
    for p in paths:
        fpath = os.path.join(p, "redtide_occurrences.csv")
        if os.path.exists(fpath):
            try:
                occur_df = pd.read_csv(fpath)
                occur_df['Date'] = pd.to_datetime(occur_df['Date'])
                # 밀도가 숫자형이 되도록 변환
                occur_df['Density'] = pd.to_numeric(occur_df['Density'], errors='coerce').fillna(0)
                break
            except: pass
            
    return env_df, occur_df

# -----------------------------------------------------------------------------
# 4. 적조 위험도 진단 로직 (사용자 지정 로직 유지)
# -----------------------------------------------------------------------------
def assess_red_tide_risk(temp, salt):
    risk_score = 0
    reasons = []

    # --- 수온 평가 ---
    if 20 == temp:
        risk_score += 70
        reasons.append("🌡️ **최적수온(20℃, 25℃, 27.5℃)**: 적조 생물 증식에 최적입니다.")
    elif 25 == temp:
        risk_score += 70
        reasons.append("🌡️ **최적수온(20℃, 25℃, 27.5℃)**: 적조 생물 증식에 최적입니다.")
    elif 27.5 == temp:
        risk_score += 70
        reasons.append("🌡️ **최적수온(20℃, 25℃, 27.5℃)**: 적조 생물 증식에 최적입니다.")
    elif 21 <= temp <= 24.9:
        risk_score += 50
        reasons.append("🌡️ **중온(21~29℃)**: 적조 생물이 양호한 성장률을 보입니다.")
    elif 25.1 <= temp <= 27.4:
        risk_score += 55
        reasons.append("🌡️ **중온(21~29℃)**: 적조 생물이 양호한 성장률을 보입니다.")
    elif 27.6 <= temp <= 30:
        risk_score += 65
        reasons.append("🌡️ **중온(21~29℃)**: 적조 생물이 양호한 성장률을 보입니다.")
    elif temp >= 30:
        risk_score -= 20
        reasons.append("🌡️ **고온(30℃↑)**: 적조 생물 성장이 확연히 저하됩니다.")
    elif temp <= 15:
        risk_score -= 20
        reasons.append("❄️ **과저수온(15℃↓)**: 적조 생물 성장이 확연히 저하됩니다.")
    else:
        risk_score -= 10
        reasons.append("🌡️ **수온**: 적조 발생 최적 범위를 벗어났습니다.")

    # --- 염분 평가 ---
    if 31 <= salt <= 34:
        risk_score += 50
        reasons.append("🧂 **염분(31~34psu)**: 적조 생물 증식에 최적입니다.")        
    elif salt <= 20:
        risk_score -= 20
        reasons.append("🧂 **저염분(20psu↓)**: 염분이 너무 낮아 적조 생물의 성장이 특히 저하됩니다.")
    else:
        risk_score -= 10
        reasons.append("🧂 **염분**: 적조 발생 최적 범위를 벗어났습니다.")

    # --- 최종 진단 ---
    if risk_score >= 85:
        return "🚨 매우 위험 (적조 대발생 가능)", "red", reasons
    elif risk_score >= 40:
        return "⚠️ 주의 (적조 발생 가능 조건 충족)", "orange", reasons
    else:
        return "✅ 안전 (적조 발생 확률 낮음)", "green", reasons

# -----------------------------------------------------------------------------
# 5. 메인 화면 구성
# -----------------------------------------------------------------------------
def main():
    st.title("🌊 통영 적조 예측 및 분석 시스템")
    st.markdown("##### 지난 25년간(2000-2024)의 통영 조위관측소 데이터 및 실제 적조 발생 이력 기반")
    
    with st.sidebar:
        st.header("데이터 현황")
        with st.spinner("데이터 로딩 중..."):
            env_df, occur_df = load_all_data()

        if env_df is not None:
            st.success("연결 성공!")
            st.metric("총 데이터", f"{len(env_df):,} 건")
            st.metric("분석 기간", f"{env_df.index.min().year} ~ {env_df.index.max().year}")
            st.info("현재 'tongyeong_lite.csv' 데이터를 사용 중입니다.")
        else:
            st.error("데이터 없음")
            st.warning("tongyeong_lite.csv 파일을 찾을 수 없습니다.")
            st.stop()
            
        if occur_df is not None:
            st.success(f"적조 발생 데이터 연결됨 ({len(occur_df):,}건)")
        else:
            st.warning("적조 발생 데이터 없음 (밀도 시각화 불가)")

    # 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs(["📅 과거 날짜 조회", "🔮 미래 날짜 예측", "🌡️ 수온별 염분 예측", "📊 데이터 분포"])

    # [탭 1] 과거 날짜 조회
    with tab1:
        st.subheader("과거 바다 상태 조회")
        col1, col2 = st.columns([1, 2])
        with col1:
            min_d, max_d = env_df.index.min().date(), env_df.index.max().date()
            # 기본값: 데이터에 존재하는 안전한 날짜
            default_d = pd.to_datetime("2005-08-18").date() 
            input_date = st.date_input("과거 날짜 선택", value=default_d, min_value=min_d, max_value=max_d)
            btn_query = st.button("조회하기", type="primary", key='btn1', use_container_width=True)

        with col2:
            if btn_query:
                target_data = env_df[env_df.index.date == input_date]
                if len(target_data) > 0:
                    avg_t, avg_s = target_data['Temp'].mean(), target_data['Salt'].mean()
                    level, color, reasons = assess_red_tide_risk(avg_t, avg_s)
                    
                    st.markdown(f"### {input_date} 분석 결과")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("수온", f"{avg_t:.2f} ℃")
                    m2.metric("염분", f"{avg_s:.2f} psu")
                    # m3.metric("위험 점수", f"{score} 점") # 함수 리턴값에 점수 없음
                    
                    st.markdown(f"#### 진단: :{color}[{level}]")
                    with st.expander("상세 진단 근거 보기", expanded=True):
                        for r in reasons: st.write(f"- {r}")
                else:
                    st.warning("해당 날짜의 데이터가 없습니다.")

    # [탭 2] 미래 날짜 예측
    with tab2:
        st.subheader("미래 시점 예측")
        st.info("과거 25년간 해당 날짜들의 평균값을 분석하여 미래의 수온과 염분을 예측합니다.")
        
        col_in, col_out = st.columns([1, 2])
        with col_in:
            # 미래 날짜는 제한 없이 선택 가능
            future_date = st.date_input("미래 날짜 선택", value=pd.to_datetime("today").date())
            btn_future = st.button("미래 예측 실행", type="primary", key='btn_future', use_container_width=True)
        
        with col_out:
            if btn_future:
                target_md = future_date.strftime('%m-%d')
                historical_samples = env_df[env_df['MM-DD'] == target_md]
                
                if len(historical_samples) > 0:
                    pred_t = historical_samples['Temp'].mean()
                    pred_s = historical_samples['Salt'].mean()
                    level, color, reasons = assess_red_tide_risk(pred_t, pred_s)
                    
                    st.markdown(f"### 🔮 {future_date} 예측 결과")
                    c1, c2 = st.columns(2)
                    c1.metric("예상 평년 수온", f"{pred_t:.2f} ℃")
                    c2.metric("예상 평년 염분", f"{pred_s:.2f} psu")
                    
                    st.markdown(f"#### 예측 진단: :{color}[{level}]")
                    st.caption(f"* 과거 {len(historical_samples)}개 연도의 {target_md} 데이터를 기반으로 분석했습니다.")
                    with st.expander("상세 진단 근거"):
                        for r in reasons: st.write(f"- {r}")
                else:
                    st.error("해당 날짜의 과거 통계 데이터가 부족합니다.")

    # [탭 3] 수온별 예측
    with tab3:
        st.subheader("수온별 염분 예측")
        col_in, col_out = st.columns([1, 2])
        with col_in:
            input_temp = st.number_input("가상 수온 입력 (℃)", value=25.5, step=0.1)
            btn_predict = st.button("예측 및 유사도 분석", type="primary", key='btn2', use_container_width=True)

        if btn_predict:
            X = env_df[['Temp']]
            y = env_df['Salt']
            model = LinearRegression()
            model.fit(X, y)
            pred_salt = model.predict([[input_temp]])[0]
            
            level, color, reasons = assess_red_tide_risk(input_temp, pred_salt)
            
            with col_out:
                st.markdown("### 1. 예측 결과")
                c1, c2 = st.columns(2)
                c1.metric("예상 염분", f"{pred_salt:.2f} psu")
                
                st.markdown(f"#### 진단: :{color}[{level}]")
                st.info("💡 **분석 근거:**\n\n" + "\n".join([f"- {r}" for r in reasons]))

                st.divider()
                
                # 유사도 확인
                st.markdown("### 2. 과거 유사 사례 (Top 5)")
                st.caption(f"수온 {input_temp}℃, 염분 {pred_salt:.2f}psu와 가장 환경이 비슷했던 과거 날짜들입니다.")
                
                env_df['Similarity'] = (env_df['Temp'] - input_temp)**2 + (env_df['Salt'] - pred_salt)**2
                top5 = env_df.sort_values('Similarity').head(5)
                st.dataframe(top5[['Temp', 'Salt']], use_container_width=True)

    # [탭 4] 데이터 시각화
    with tab4:
        st.subheader("통영 해역 수온·염분 분포")
        
        if st.checkbox("그래프 보기", value=True):
            fig, ax = plt.subplots(figsize=(10, 6))
            
    # 1. 데이터 병합 및 전처리 (Data Preparation)
    # (1) 배경 데이터 준비: Density 컬럼을 추가하고 0으로 설정
    bg_sample = env_df.sample(min(len(env_df), 5000)).copy()    
    bg_sample['Density'] = 0  # 적조 없음 = 밀도 0

    # (2) 적조 발생 데이터 준비
    if occur_df is not None and not occur_df.empty:
        target_df = occur_df[occur_df['Density'] > 0].copy()
    else:
        target_df = pd.DataFrame(columns=bg_sample.columns)

    # (3) 데이터 합치기 (Concatenate)
    # 배경과 적조 데이터를 하나로 합칩니다.
    total_df = pd.concat([bg_sample, target_df], ignore_index=True)

    # (4) 정렬 (Sorting)
    # 밀도가 낮은 점(0/회색)이 밑에 깔리고, 높은 점(빨강)이 위에 오도록 정렬 (중요)
    total_df = total_df.sort_values('Density', ascending=True)

    # 2. 시각화 설정 (Custom Visualization)
    base_cmap = plt.cm.get_cmap('Reds')
    colors = [base_cmap(i) for i in range(base_cmap.N)]
    colors[0] = mcolors.to_rgba('lightgrey') # 0값(가장 낮은 값)을 회색으로 강제 설정
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('GreyRed', colors, base_cmap.N)

    # (2) 사이즈 계산용 컬럼 (로그 스케일 적용)
    total_df['Size_Scale'] = np.log1p(total_df['Density']) * 10

    # 3. 플롯 그리기 (Plotting)
    points = sns.scatterplot(
        data=total_df,
        x='Temp',
        y='Salt',
        hue='Density',
        size='Size_Scale',
        sizes=(20, 300),
        palette=custom_cmap,
        edgecolor='black',
        linewidth=0.3,
        alpha=0.7,
        ax=ax,
        legend=False
    )

    # 4. 컬러바 추가 (Colorbar)
    norm = plt.Normalize(vmin=0, vmax=total_df['Density'].max())
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Red Tide Density (cells/mL) \n(Grey=0, Red=High)', rotation=270, labelpad=20)
    

            # 위험 구간 박스
import matplotlib.patches as patches
            rect = patches.Rectangle((23, 30), 5, 4, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            ax.set_xlabel("Temp (℃)")
            ax.set_ylabel("Salt (psu)")
            ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

if __name__ == "__main__":
    main()



















