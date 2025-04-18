import numpy as np
import pandas as pd
import datetime
import scipy
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import r_regression
from sklearn.linear_model import LinearRegression, Lasso, Ridge

# RandomForestRegressor를 임포트합니다.
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pickle
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

if platform.system() == "Darwin":  #
    plt.rc("font", family="AppleGothic")
else:
    plt.rc("font", family="NanumGothic")

fe = fm.FontEntry(
    fname=r"/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # ttf 파일이 저장되어 있는 경로
    name="NanumGothic",
)  # 원하는 폰트 설정
fm.fontManager.ttflist.insert(0, fe)  # Matplotlib에 폰트 추가

plt.rcParams.update({"font.size": 18, "font.family": "NanumGothic"})  # 폰트 설정

plt.rcParams["axes.unicode_minus"] = False

# 데이터 로드
def data_load(path):
    df = pd.read_csv(path)
    df = df.set_index("시간")
    df.index = pd.DatetimeIndex(load_df.index)
    return df

raw_df = data_load("./data/SN_total.csv")


# 변수 로그화
def log_transform(df, columns) : 
    for col in columns:
        df[col] = np.log10(df[col])
    return df

raw_df = log_transform(raw_df,["로그 원수 탁도", '로그 응집제 주입률'])

# 변수 선택

X = df[
    [
        "로그 원수 탁도",
        "원수 pH",
        "원수 알칼리도",
        "원수 전기전도도",
        "원수 수온",
        "3단계 원수 유입 유량",
        "3단계 침전지 체류시간",
    ]
]
y = df["로그 응집제 주입률"]
Xt, Xts, yt, yts = train_test_split(X, y, test_size=0.2, shuffle=False)    

# xgboost 모델

from xgboost import XGBRegressor

regressor = XGBRegressor(random_state=2, n_jobs=-1)
