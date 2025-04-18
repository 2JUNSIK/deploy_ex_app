import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# 데이터 로드 함수
def data_load(path):
    df = pd.read_csv(path)
    df = df.set_index("시간")
    df.index = pd.DatetimeIndex(df.index)
    return df

# 로그 변환 함수
def log_transform(df, columns):
    for col in columns:
        df[col] = np.log10(df[col])
    return df

# Streamlit 앱
st.title("XGBoost 예측 모델 시각화 앱")
st.markdown("원수 데이터로 응집제 주입률을 예측하고, 입력 변수와 모델 파라미터를 조정해 결과를 확인하세요.")

# 데이터 로드
try:
    df = data_load("./data/SN_total.csv")
    df = log_transform(df, ["원수 탁도", "3단계 1계열 응집제 주입률"])
except FileNotFoundError:
    st.error("데이터 파일(SN_total.csv)을 ./data/ 폴더에 추가하세요.")
    st.stop()

# 입력 변수 선택
st.sidebar.header("모델 입력 설정")
available_columns = [
    "원수 탁도",
    "원수 pH",
    "원수 알칼리도",
    "원수 전기전도도",
    "원수 수온",
    "3단계 원수 유입 유량"
]
selected_columns = st.sidebar.multiselect(
    "입력 변수 선택", available_columns, default=["원수 탁도"]
)

# 파라미터 선택
st.sidebar.header("XGBoost 파라미터")
max_depth = st.sidebar.selectbox("max_depth", [2, 3, 4, 5], index=0)
n_estimators = st.sidebar.selectbox("n_estimators", [50, 100, 200], index=1)
eta = st.sidebar.selectbox("eta", [0.01, 0.02, 0.05, 0.1], index=1)
subsample = st.sidebar.selectbox("subsample", [0.6, 0.8, 1.0], index=1)
min_child_weight = st.sidebar.selectbox("min_child_weight", [1, 3, 5], index=0)

# 파라미터 딕셔너리
params = {
    "max_depth": max_depth,
    "n_estimators": n_estimators,
    "eta": eta,
    "subsample": subsample,
    "min_child_weight": min_child_weight,
}

# 데이터 준비
if not selected_columns:
    st.warning("최소 하나의 입력 변수를 선택하세요.")
    st.stop()

X = df[selected_columns]
y = df["3단계 1계열 응집제 주입률"]
Xt, Xts, yt, yts = train_test_split(X, y, test_size=0.2, shuffle=False)

# 모델 학습
regressor = XGBRegressor(random_state=2, n_jobs=-1)
model = regressor.set_params(**params)
model.fit(Xt, yt)

# 예측
yt_pred = model.predict(Xt)
yts_pred = model.predict(Xts)

# 메트릭 계산
mse_train = mean_squared_error(10**yt, 10**yt_pred)
mse_test = mean_squared_error(10**yts, 10**yts_pred)
r2_train = r2_score(10**yt, 10**yt_pred)
r2_test = r2_score(10**yts, 10**yts_pred)

# 메트릭 출력
st.subheader("모델 성능 메트릭")
col1, col2 = st.columns(2)
col1.metric("학습 데이터 MSE", f"{mse_train:.4f}")
col1.metric("학습 데이터 R²", f"{r2_train:.2f}")
col2.metric("테스트 데이터 MSE", f"{mse_test:.4f}")
col2.metric("테스트 데이터 R²", f"{r2_test:.2f}")

# 시각화
st.subheader("예측 결과 시각화")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 학습 데이터 산점도
ax = axes[0]
ax.scatter(Xt[selected_columns[0]], yt, s=3, label="실제")
ax.scatter(Xt[selected_columns[0]], yt_pred, s=3, label="예측", c="r")
ax.grid()
ax.legend(fontsize=10)
ax.set_xlabel(selected_columns[0])
ax.set_ylabel("로그 응집제 주입률")
ax.set_title(f"학습 데이터\nMSE: {mse_train:.4f}, R²: {r2_train:.2f}")

# 테스트 데이터 산점도
ax = axes[1]
ax.scatter(Xts[selected_columns[0]], yts, s=3, label="실제")
ax.scatter(Xts[selected_columns[0]], yts_pred, s=3, label="예측", c="r")
ax.grid()
ax.legend(fontsize=10)
ax.set_xlabel(selected_columns[0])
ax.set_ylabel("로그 응집제 주입률")
ax.set_title(f"테스트 데이터\nMSE: {mse_test:.4f}, R²: {r2_test:.2f}")

plt.tight_layout()
st.pyplot(fig)

# 데이터 파일 경고
st.sidebar.info("데이터 파일은 './data/SN_total.csv' 경로에 있어야 합니다.")