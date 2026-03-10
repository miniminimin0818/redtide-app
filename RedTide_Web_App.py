import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
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
                
                rename_dict = {
                    '수온': 'Temp', 
                    '염분': 'Salt', 
                    '풍향': 'WindDir', 
                    '풍속': 'WindSpeed', 
                    '조위': 'Tide'
                }
                env_df.rename(columns=rename_dict, inplace=True)
                
                env_df['Date'] = pd.to_datetime(env_df['Date'])
                env_df.set_index('Date', inplace=True)
                
                # 이상치 제거
                if 'Temp' in env_df.columns and 'Salt' in env_df.columns:
                    env_df = env_df[(env_df['Temp'] > 0) & (env_df['Salt'] > 0) & (env_df['Salt'] < 45)]
                
                # 월-일 정보 추가
                env_df['MM-DD'] = env_df.index.strftime('%m-%d')
                break
            except Exception as e:
                # 에러를 숨기지 않고 터미널이나 화면에 출력하여 디버깅을 돕습니다.
                print(f"tongyeong_lite.csv 로드 중 에러: {e}")
                pass
            
    # 2. 적조 발생 데이터 로드
    for p in paths:
        fpath = os.path.join(p, "redtide_occurrences.csv")
        if os.path.exists(fpath):
            try:
                occur_df = pd.read_csv(fpath)
                occur_df['Date'] = pd.to_datetime(occur_df['Date'])
                occur_df['Density'] = pd.to_numeric(occur_df['Density'], errors='coerce').fillna(0)
                break
            except Exception as e:
                print(f"redtide_occurrences.csv 로드 중 에러: {e}")
                pass
            
    return env_df, occur_df
    
# -----------------------------------------------------------------------------
# 4. 적조 위험도 진단 로직 (5가지 변수 적용)
# -----------------------------------------------------------------------------
def assess_red_tide_risk(temp, salt, wind_dir, wind_speed, tide):
    risk_score = 0
    reasons = []

    # --- 수온 평가 ---
    if 20 <= temp <= 27.5:
        risk_score += 40
        reasons.append(f"🌡️ 수온({temp:.1f}℃): 적조 생물 증식 최적 범위입니다.")
    else:
        risk_score -= 10
        reasons.append(f"🌡️ 수온({temp:.1f}℃): 증식 최적 범위를 벗어났습니다.")

    # --- 염분 평가 ---
    if 31 <= salt <= 34:
        risk_score += 20
        reasons.append(f"🧂 염분({salt:.1f}psu): 성장에 적합한 염분입니다.")
    else:
        risk_score -= 10
        reasons.append(f"🧂 염분({salt:.1f}psu): 적조 발생 확률이 낮은 염분대입니다.")

    # --- 풍속 평가 ---
    if wind_speed < 4.0:
        risk_score += 15
        reasons.append(f"🌬️ 풍속({wind_speed:.1f}m/s): 바람이 약해 해수면이 성층화되어 집적에 유리합니다.")
    elif wind_speed > 8.0:
        risk_score -= 15
        reasons.append(f"🌀 풍속({wind_speed:.1f}m/s): 강풍으로 해수가 혼합되어 적조가 분산됩니다.")

    # --- 풍향 평가 (135~225도: 남풍 계열) ---
    if 135 <= wind_dir <= 225:
        risk_score += 15
        reasons.append(f"🧭 풍향({wind_dir:.1f}º): 남풍 계열로 외해의 적조가 연안으로 밀려올 위험이 큽니다.")
    elif 315 <= wind_dir or wind_dir <= 45:
        risk_score -= 10
        reasons.append(f"🧭 풍향({wind_dir:.1f}º): 북풍 계열로 적조가 외해로 흩어지기 쉽습니다.")

    # --- 조위 평가 ---
    reasons.append(f"🌊 조위({tide:.1f}cm): 현재 조위 상태입니다.")

    # --- 최종 진단 ---
    if risk_score >= 70:
        return "🚨 매우 위험", "red", reasons
    elif risk_score >= 40:
        return "⚠️ 주의", "orange", reasons
    else:
        return "✅ 안전", "green", reasons
# -----------------------------------------------------------------------------
# 5. 메인 화면 구성
# -----------------------------------------------------------------------------
def main():
    st.title("🌊 통영 적조 예측 및 분석 시스템")
    st.markdown("##### 지난 25년간(2000-2024)의 통영 조위관측소 데이터 및 실제 적조 발생 이력 기반")
    
    # 💡 [핵심 해결 포인트] 변수를 미리 None으로 초기화하여 NameError를 원천 차단합니다.
    ml_data = None 
    
    with st.sidebar:
        st.header("데이터 현황")
        with st.spinner("데이터 로딩 중..."):
            env_df, occur_df = load_all_data()

        if env_df is not None:
            st.success("환경 데이터 연결 성공!")
            st.metric("총 데이터", f"{len(env_df):,} 건")
        else:
            st.error("데이터 없음")
            st.warning("tongyeong_lite.csv 파일을 찾을 수 없습니다.")
            st.stop()
            
        if occur_df is not None:
            st.success(f"적조 발생 데이터 연결됨 ({len(occur_df):,}건)")
            ml_data = pd.merge(env_df.reset_index(), occur_df[['Date', 'Density']], on='Date', how='left')
            ml_data['Density'] = ml_data['Density'].fillna(0)
            
            # 💡 [필수 방어 코드] 컬럼 존재 여부 확인
            req_cols = ['Temp', 'Salt', 'WindDir', 'WindSpeed', 'Tide']
            missing_cols = [col for col in req_cols if col not in ml_data.columns]
            
            if missing_cols:
                st.error(f"🚨 부족한 컬럼이 있어 ML을 켤 수 없습니다: {missing_cols}")
                ml_data = None
            else:
                ml_data = ml_data.dropna(subset=req_cols)
        else:
            st.warning("적조 발생 데이터 없음 (머신러닝 시각화 불가)")

    # 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs(["📅 과거 날짜 조회", "🔮 미래 날짜 예측", "🌡️ 수온별 염분 예측", "📊 데이터 분포"])

    # [탭 1] 과거 날짜 조회
    with tab1:
        st.subheader("과거 바다 상태 조회")
        col1, col2 = st.columns([1, 2])
        with col1:
            min_d, max_d = env_df.index.min().date(), env_df.index.max().date()
            
            # 기본값 설정: 2005-08-18이 데이터 범위 내에 있는지 확인
            target_default = pd.to_datetime("2005-08-18").date()
            if min_d <= target_default <= max_d:
                default_d = target_default
            else:
                # 범위를 벗어나면 데이터의 가장 최신 날짜(max_d)를 기본값으로 사용
                default_d = max_d 
                
            input_date = st.date_input("과거 날짜 선택", value=default_d, min_value=min_d, max_value=max_d)
            btn_query = st.button("조회하기", type="primary", use_container_width=True)
            
        with col2:
            if btn_query:
                target_data = env_df[env_df.index.date == input_date]
                if len(target_data) > 0:
                    # 5개 변수 평균 계산
                    t, s, wd, ws, td = target_data[['Temp', 'Salt', 'WindDir', 'WindSpeed', 'Tide']].mean()
                    level, color, reasons = assess_red_tide_risk(t, s, wd, ws, td)
                    
                    st.markdown(f"### {input_date} 분석 결과")
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("수온", f"{t:.2f} ℃")
                    m2.metric("염분", f"{s:.2f} psu")
                    m3.metric("풍향", f"{wd:.1f} º")
                    m4.metric("풍속", f"{ws:.1f} m/s")
                    m5.metric("조위", f"{td:.0f} cm")
                    
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
                    # 5개 변수 평년값 계산
                    t, s, wd, ws, td = historical_samples[['Temp', 'Salt', 'WindDir', 'WindSpeed', 'Tide']].mean()
                    level, color, reasons = assess_red_tide_risk(t, s, wd, ws, td)
                    
                    st.markdown(f"### 🔮 {future_date} 예측 결과")
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("예상 수온", f"{t:.2f} ℃")
                    c2.metric("예상 염분", f"{s:.2f} psu")
                    c3.metric("예상 풍향", f"{wd:.1f} º")
                    c4.metric("예상 풍속", f"{ws:.1f} m/s")
                    c5.metric("예상 조위", f"{td:.0f} cm")
                    
                    st.markdown(f"#### 예측 진단: :{color}[{level}]")
                    st.caption(f"* 과거 {len(historical_samples)}개 연도의 {target_md} 데이터를 기반으로 분석했습니다.")
                    with st.expander("상세 진단 근거"):
                        for r in reasons: st.write(f"- {r}")
                else:
                    st.error("해당 날짜의 과거 통계 데이터가 부족합니다.")

    # [탭 3] 다변수 시뮬레이션
    with tab3:
        st.subheader("다변수 선형회귀(Linear Regression) 기반 적조 발생 시뮬레이션")
        st.info("5가지 환경 변수를 조절하여 머신러닝 모델이 예측하는 '적조 발생 밀도'와 각 요인의 영향력을 확인하세요.")
        
        if ml_data is not None:
            # 모델 학습
            features = ['Temp', 'Salt', 'WindDir', 'WindSpeed', 'Tide']
            X = ml_data[features]
            y = ml_data['Density']
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            lr_model = LinearRegression()
            lr_model.fit(X_scaled, y)
            
            col_in, col_out = st.columns([1, 2])
            with col_in:
                st.markdown("해양 환경 가상 설정")
                in_t = st.slider("수온 (℃)", 15.0, 32.0, 25.0, 0.1)
                in_s = st.slider("염분 (psu)", 20.0, 36.0, 32.0, 0.1)
                in_wd = st.slider("풍향 (º, 180=남풍)", 0.0, 360.0, 180.0, 1.0)
                in_ws = st.slider("풍속 (m/s)", 0.0, 20.0, 3.0, 0.1)
                in_td = st.slider("조위 (cm)", 0.0, 400.0, 150.0, 1.0)
                
            with col_out:
                # 사용자가 슬라이더로 입력한 값을 스케일링 후 예측
                input_scaled = scaler.transform([[in_t, in_s, in_wd, in_ws, in_td]])
                pred_density = lr_model.predict(input_scaled)[0]
                pred_density = max(0, pred_density) # 예측값이 음수면 0으로 처리
                
                st.markdown("AI 예측 적조 밀도")
                st.metric(label="예상 Red Tide Density", value=f"{pred_density:,.0f} cells/mL")
                
                if pred_density > 1000:
                    st.error("🚨 경고: 적조 주의보 발령 기준(1,000 cells/mL)을 초과하는 환경입니다!")
                elif pred_density > 0:
                    st.warning("⚠️ 주의: 적조 발생 가능성이 있는 환경입니다.")
                else:
                    st.success("✅ 안전: 적조가 발생하기 어려운 환경입니다.")

                st.divider()
                st.markdown("분석 변수별 영향력 도출")
                
                coef_dict = dict(zip(features, lr_model.coef_))
                for feature, coef in coef_dict.items():
                    direction = "증가" if coef > 0 else "감소"
                    color = "red" if coef > 0 else "blue"
                    st.write(f"- **{feature}**: 수치가 커질수록 적조 밀도가 :{color}[**{direction}**]하는 경향성 (가중치: {coef:.2f})")
        else:
            st.error("적조 발생 데이터가 없어 머신러닝 학습을 진행할 수 없습니다.")
    
    # [탭 4] 데이터 시각화
    with tab4:
        st.subheader("통영 해역 수온·염분 분포와 적조 밀도")
        
        if st.checkbox("그래프 보기", value=True):
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # 1. 데이터 병합
            bg_sample = env_df.sample(min(len(env_df), 5000)).copy()    
            bg_sample['Density'] = 0  
            # 💡 [핵심] occur_df가 아닌 수온/염분/밀도가 모두 모여있는 ml_data를 사용!
            if ml_data is not None and not ml_data.empty:
                target_df = ml_data[ml_data['Density'] > 0].copy()
            else:
                target_df = pd.DataFrame(columns=bg_sample.columns)

            # 2. 시각화 스타일 설정
            base_cmap = plt.cm.get_cmap('Reds')
            colors_list = [base_cmap(i) for i in range(base_cmap.N)]
            colors_list[0] = mcolors.to_rgba('white')
            custom_cmap = mcolors.LinearSegmentedColormap.from_list('WhiteRed', colors_list, base_cmap.N)
            total_df['Size_Scale'] = np.log1p(total_df['Density']) 

            # 3. 플롯 그리기
            points = sns.scatterplot(
                data=total_df,
                x='Temp',
                y='Salt',
                hue='Density',
                size='Size_Scale',
                sizes=(30, 650),
                palette=custom_cmap,
                edgecolor='black',
                linewidth=0.5,
                alpha=0.7,
                ax=ax,
                legend=False
            )
            
            # 4. 부가 요소 (컬러바, 위험구역 박스)
            norm = plt.Normalize(vmin=0, vmax=total_df['Density'].max())
            sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Red Tide Density (cells/mL)', rotation=270, labelpad=20)
            
            # 위험 구간 박스 (수온 23-28, 염분 30-34 부근 강조)
            import matplotlib.patches as patches
            rect = patches.Rectangle((22, 30), 4, 4, linewidth=3, edgecolor='red', facecolor='none', linestyle='--')
            ax.add_patch(rect)
            ax.text(22, 28, "Red Tide Optimum", color='red', fontsize=11.5, fontweight='bold')
            
            ax.set_xlabel("Temp (℃)")
            ax.set_ylabel("Salt (psu)")
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)

if __name__ == "__main__":

    main()


