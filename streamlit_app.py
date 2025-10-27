# app.py
# -------------------------------------------------------------
# Streamlit Math Mastery App — Final MVP (with latest requests)
# Author: ChatGPT
# -------------------------------------------------------------

import os
import json
import time
import uuid
import random
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

# =============== 기본 설정 ===============
st.set_page_config(page_title="개념 마스터 (MVP)", layout="wide")

APP_TITLE = "개념 마스터 (MVP)"
DATA_DIR = "data"
RESPONSES_CSV = os.path.join(DATA_DIR, "responses.csv")
USERS_CSV = os.path.join(DATA_DIR, "users.csv")
ITEMS_CSV = os.path.join(DATA_DIR, "items.csv")  # 선택: 외부 아이템 업로드용

os.makedirs(DATA_DIR, exist_ok=True)

# =============== 유틸 & 스키마 ===============
def _now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

REQUIRED_USER_COLS = ["user_id","user_name","role","grade","age","created_at"]

@st.cache_data(show_spinner=False)
def _empty_users_df():
    return pd.DataFrame(columns=REQUIRED_USER_COLS)

def load_users_df() -> pd.DataFrame:
    """Load users.csv and migrate columns if file exists with older schema."""
    if os.path.exists(USERS_CSV):
        try:
            df = pd.read_csv(USERS_CSV)
        except Exception:
            df = _empty_users_df()
    else:
        df = _empty_users_df()
    for c in REQUIRED_USER_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[REQUIRED_USER_COLS]
    return df

# 최초 파일 생성
if not os.path.exists(RESPONSES_CSV):
    pd.DataFrame(columns=[
        "ts","user_id","user_name","role","area","subtopic","item_id","is_correct",
        "response","response_time","error_tag","level","attempt_id"
    ]).to_csv(RESPONSES_CSV, index=False, encoding="utf-8-sig")

if not os.path.exists(USERS_CSV):
    _empty_users_df().to_csv(USERS_CSV, index=False, encoding="utf-8-sig")

# =============== 세션 상태 ===============
if "user" not in st.session_state:
    st.session_state.user = {"user_id": None, "user_name": None, "role": "학생", "grade": None, "age": None}
if "quiz" not in st.session_state:
    st.session_state.quiz = {
        "pool": [], "current_idx": 0, "start_ts": None, "attempt_id": None,
        "area": None, "subtopics": [], "levels": ["L1","L2","L3"], "size": 8
    }
if "responses" not in st.session_state:
    st.session_state.responses = []

# =============== 시드 문항 ===============
@st.cache_data(show_spinner=False)
def load_seed_items() -> pd.DataFrame:
    seed = [
        # 대수
        {"item_id":"ALG-001","area":"대수","subtopic":"다항식 전개","level":"L1","time_hint":30,
         "stem":"(x+2)(x-3)를 전개하시오.","choices":None,"answer":"x^2 - x - 6",
         "explanation":"분배법칙: x*(x-3)+2*(x-3)=x^2-3x+2x-6=x^2-x-6",
         "error_tags":["절차오류","계산실수"]},
        {"item_id":"ALG-002","area":"대수","subtopic":"인수분해","level":"L1","time_hint":35,
         "stem":"x^2-5x+6을 인수분해 하시오.","choices":None,"answer":"(x-2)(x-3)",
         "explanation":"근의 합 5, 곱 6 → 2와 3","error_tags":["개념미이해"]},
        {"item_id":"ALG-003","area":"대수","subtopic":"인수분해","level":"L2","time_hint":45,
         "stem":"x^2+7x+10의 두 근을 구하시오.","choices":None,"answer":"-5, -2",
         "explanation":"곱이 10, 합이 7 → (-5),(-2)","error_tags":["개념미이해","계산실수"]},
        {"item_id":"ALG-004","area":"대수","subtopic":"등식의 변형","level":"L2","time_hint":40,
         "stem":"2x+5=19일 때 x를 구하시오.","choices":None,"answer":"7",
         "explanation":"2x=14 → x=7","error_tags":["절차오류"]},
        {"item_id":"ALG-005","area":"대수","subtopic":"연립방정식","level":"L2","time_hint":60,
         "stem":"x+y=7, x-y=1을 풀어 x,y를 구하시오.","choices":None,"answer":"(4,3)",
         "explanation":"가감법으로 x=4,y=3","error_tags":["절차오류","계산실수"]},
        {"item_id":"ALG-006","area":"대수","subtopic":"지수법칙","level":"L1","time_hint":40,
         "stem":"a^2·a^3 = ?","choices":None,"answer":"a^5",
         "explanation":"지수 더하기","error_tags":["개념미이해"]},
        # 함수
        {"item_id":"FUN-001","area":"함수","subtopic":"함수 개념","level":"L1","time_hint":40,
         "stem":"y=2x+1에서 x=3일 때 y의 값은?","choices":None,"answer":"7",
         "explanation":"대입 계산","error_tags":["계산실수"]},
        {"item_id":"FUN-002","area":"함수","subtopic":"일차함수 그래프","level":"L2","time_hint":60,
         "stem":"y=3x-2의 그래프의 기울기는?","choices":["-2","0","3","2/3"],"answer":"3",
         "explanation":"y=mx+b에서 m=3","error_tags":["개념미이해"]},
        {"item_id":"FUN-003","area":"함수","subtopic":"함수와 대응","level":"L2","time_hint":60,
         "stem":"다음 중 함수가 아닌 것은?","choices":["x→x^2","x→|x|","원점대칭","원의 방정식 y=±√(r^2-x^2)"],
         "answer":"원의 방정식 y=±√(r^2-x^2)",
         "explanation":"x 하나에 y 두 개 → 대응 불가","error_tags":["개념미이해","문제해석"]},
        {"item_id":"FUN-004","area":"함수","subtopic":"최대최소","level":"L3","time_hint":75,
         "stem":"함수 f(x)=x^2-4x+5의 최솟값은?","choices":None,"answer":"1",
         "explanation":"완전제곱식 (x-2)^2+1 → 최솟값 1","error_tags":["개념미이해"]},
        {"item_id":"FUN-005","area":"함수","subtopic":"함수합성","level":"L3","time_hint":80,
         "stem":"f(x)=2x, g(x)=x+3일 때 (f∘g)(2)의 값은?","choices":None,"answer":"10",
         "explanation":"g(2)=5, f(5)=10","error_tags":["절차오류","계산실수"]},
        # 기하
        {"item_id":"GEO-001","area":"기하","subtopic":"삼각형 성질","level":"L1","time_hint":45,
         "stem":"삼각형의 내각의 합은?","choices":["90°","120°","180°","360°"],"answer":"180°",
         "explanation":"기본 성질","error_tags":["개념미이해"]},
        {"item_id":"GEO-002","area":"기하","subtopic":"피타고라스","level":"L1","time_hint":45,
         "stem":"직각삼각형에서 빗변이 13, 한 변이 5일 때 다른 변은?","choices":None,"answer":"12",
         "explanation":"13^2-5^2=169-25=144 → 12","error_tags":["계산실수"]},
        {"item_id":"GEO-003","area":"기하","subtopic":"닮음","level":"L2","time_hint":60,
         "stem":"닮음비가 2:3인 두 도형의 넓이비는?","choices":None,"answer":"4:9",
         "explanation":"넓이비 = 선분비^2","error_tags":["개념미이해"]},
        {"item_id":"GEO-004","area":"기하","subtopic":"삼각비","level":"L2","time_hint":60,
         "stem":"sin30°, cos60°, tan45°를 각각 쓰시오.","choices":None,"answer":"1/2, 1/2, 1",
         "explanation":"표준각 삼각비","error_tags":["개념미이해","계산실수"]},
        # 확률과 통계
        {"item_id":"STA-001","area":"확률과 통계","subtopic":"경우의 수","level":"L1","time_hint":45,
         "stem":"동전을 두 번 던질 때 나올 수 있는 경우의 수는?","choices":None,"answer":"4",
         "explanation":"HH, HT, TH, TT","error_tags":["개념미이해"]},
        {"item_id":"STA-002","area":"확률과 통계","subtopic":"확률","level":"L1","time_hint":45,
         "stem":"공정한 주사위 한 번의 6이 나올 확률은?","choices":None,"answer":"1/6",
         "explanation":"기본 확률","error_tags":["개념미이해"]},
        {"item_id":"STA-003","area":"확률과 통계","subtopic":"평균","level":"L1","time_hint":45,
         "stem":"데이터 2,4,6,8의 평균은?","choices":None,"answer":"5",
         "explanation":"(2+4+6+8)/4=20/4=5","error_tags":["계산실수"]},
        {"item_id":"STA-004","area":"확률과 통계","subtopic":"표준편차","level":"L2","time_hint":70,
         "stem":"데이터 1,3,5의 표준편차(모표준편차 기준)를 구하시오.","choices":None,"answer":"~1.632",
         "explanation":"평균3, 분산[(4+0+4)/3]=8/3 → 표준편차≈√(2.666)=1.632","error_tags":["계산실수"]},
    ]
    return pd.DataFrame(seed)

@st.cache_data(show_spinner=False)
def load_items_from_csv(uploaded: pd.DataFrame | None) -> pd.DataFrame:
    if uploaded is not None:
        required = {"item_id","area","subtopic","level","time_hint","stem","answer"}
        if not required.issubset(set(uploaded.columns)):
            st.warning("CSV 컬럼이 부족합니다. 시드 문항을 사용합니다.")
            return load_seed_items()
        return uploaded
    return load_seed_items()

# =============== 상단/사이드바 ===============
with st.sidebar:
    st.header("로그인")
    user_name = st.text_input("이름(혹은 별칭)")
    grade = st.selectbox("학년(선택)", ["선택안함","중1","중2","중3","고1","고2","고3"], index=0)
    age_str = st.text_input("나이(선택, 숫자)", "")
    if st.button("확인/저장", use_container_width=True):
        if not user_name:
            st.error("이름을 입력하세요.")
        else:
            role = "학생"  # 고정
            uid = st.session_state.user.get("user_id") or str(uuid.uuid4())
            age_val = int(age_str) if age_str.isdigit() else None
            grade_val = None if grade == "선택안함" else grade
            st.session_state.user = {"user_id": uid, "user_name": user_name, "role": role, "grade": grade_val, "age": age_val}
            users_df = load_users_df()
            new_row = {
                "user_id": uid,
                "user_name": user_name,
                "role": role,
                "grade": grade_val,
                "age": age_val,
                "created_at": _now_str(),
            }
            users_df = pd.concat([users_df, pd.DataFrame([new_row])], ignore_index=True)
            users_df.to_csv(USERS_CSV, index=False, encoding="utf-8-sig")
            st.success(f"환영합니다, {user_name} (학생)")

st.title(APP_TITLE)
user = st.session_state.user
if user["user_name"]:
    extra = []
    if user.get("grade"): extra.append(f"학년: {user['grade']}")
    if user.get("age") is not None: extra.append(f"나이: {user['age']}")
    extra_str = " · ".join(extra)
    st.caption(f"접속: {user['user_name']} · 역할: 학생" + (f" · {extra_str}" if extra_str else ""))

# =============== 탭 구성 ===============
TABS = st.tabs(["퀴즈", "결과/보강", "재평가", "교사 대시보드", "문항 업로드"])

# =============== 문항 업로드 탭 ===============
with TABS[4]:
    st.subheader("문항 업로드 (선택)")
    st.write("CSV 컬럼 예시: item_id, area, subtopic, level, time_hint, stem, choices(json옵션), answer, explanation, error_tags(json옵션)")
    uploaded_file = st.file_uploader("CSV 업로드", type=["csv"])
    uploaded_df = None
    if uploaded_file:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.success(f"업로드 성공: {uploaded_df.shape}")
            st.dataframe(uploaded_df.head())
        except Exception as e:
            st.error(f"업로드 실패: {e}")

items_df = load_items_from_csv(uploaded_df)
for col in ["choices","error_tags"]:
    if col in items_df.columns:
        items_df[col] = items_df[col].apply(lambda x: json.loads(x) if isinstance(x, str) and x.strip().startswith("[") else x)

# =============== 유틸: 퀴즈 풀 생성 ===============
def build_quiz_pool(df: pd.DataFrame, area: str, levels: List[str], size: int) -> List[Dict[str,Any]]:
    subset = df[(df["area"]==area) & (df["level"].isin(levels))].copy()
    if subset.empty:
        return []
    pool = subset.sample(n=min(size, len(subset)), replace=False, random_state=random.randint(0, 9999))
    return pool.to_dict(orient="records")

# =============== 탭1: 퀴즈 ===============
with TABS[0]:
    st.subheader("영역별 퀴즈")

    cols = st.columns([1,1,1,1,1])
    with cols[0]:
        area = st.selectbox("영역", sorted(items_df["area"].unique().tolist()))
    with cols[1]:
        levels = st.multiselect("난이도", ["L1","L2","L3"], default=["L1","L2","L3"])
    with cols[2]:
        size = st.number_input("문항 수", min_value=3, max_value=20, value=8, step=1)
    with cols[3]:
        include_timer = st.toggle("반응시간 기록", value=True)
    with cols[4]:
        start_btn = st.button("퀴즈 시작/다시 만들기", use_container_width=True)

    if start_btn:
        st.session_state.quiz.update({
            "pool": build_quiz_pool(items_df, area, levels, size),
            "current_idx": 0,
            "start_ts": time.time() if include_timer else None,
            "attempt_id": str(uuid.uuid4()),
            "area": area,
            "subtopics": [],
            "levels": levels,
            "size": size,
        })
        st.success(f"퀴즈 생성: {len(st.session_state.quiz['pool'])}문항")

    quiz = st.session_state.quiz
    if quiz["pool"]:
        q = quiz["pool"][quiz["current_idx"]]
        st.markdown(f"#### Q{quiz['current_idx']+1}. {q['stem']}")
        # 보기/입력
        if q.get("choices"):
            user_answer = st.radio("정답 선택", q["choices"], index=None, key=f"choice_{quiz['attempt_id']}_{quiz['current_idx']}")
        else:
            user_answer = st.text_input("답 입력 (수식/숫자/문자열)", key=f"text_{quiz['attempt_id']}_{quiz['current_idx']}")

        err_tag = st.selectbox("(선택) 내가 생각하는 오류 유형", ["선택안함","개념미이해","절차오류","계산실수","문제해석","시간관리"])
        submit = st.button("제출", type="primary")

        if submit:
            ans = str(user_answer).strip()
            gold = str(q["answer"]).strip()
            norm = lambda s: s.replace(" ", "").lower()
            is_correct = norm(ans) == norm(gold)

            resp_time = None
            if include_timer and quiz["start_ts"]:
                resp_time = round(time.time() - quiz["start_ts"], 2)
                quiz["start_ts"] = time.time()

            user_id = user.get("user_id") or "anon"
            user_name = user.get("user_name") or "anon"
            role = user.get("role") or "학생"
            row = {
                "ts": _now_str(),
                "user_id": user_id,
                "user_name": user_name,
                "role": role,
                "area": q["area"],
                "subtopic": q["subtopic"],
                "item_id": q["item_id"],
                "is_correct": int(is_correct),
                "response": ans,
                "response_time": resp_time,
                "error_tag": None if err_tag=="선택안함" else err_tag,
                "level": q["level"],
                "attempt_id": quiz["attempt_id"],
            }

            st.session_state.responses.append(row)

            try:
                old = pd.read_csv(RESPONSES_CSV)
                old.loc[len(old)] = row
                old.to_csv(RESPONSES_CSV, index=False, encoding="utf-8-sig")
            except Exception as e:
                st.warning(f"로컬 저장 실패(세션에는 저장됨): {e}")

            if is_correct:
                st.success("정답입니다! 🎉")
            else:
                st.error("오답입니다.")
                with st.expander("해설 보기"):
                    st.write(q.get("explanation","(해설 준비중)"))

            if quiz["current_idx"] < len(quiz["pool"]) - 1:
                quiz["current_idx"] += 1
                st.rerun()
            else:
                st.success("퀴즈 종료! 결과/보강 탭 또는 아래에서 전체 해설을 확인하세요.")
                # 전체 해설: 이번 시도 전체 문항 요약
                with st.expander("📚 이번 세트 전체 해설 보기", expanded=True):
                    resp_list = st.session_state.get("responses", [])
                    attempt_id = quiz["attempt_id"]
                    resp_map = {r["item_id"]: r for r in resp_list if r.get("attempt_id") == attempt_id}
                    for i, itm in enumerate(quiz["pool"], start=1):
                        r = resp_map.get(itm["item_id"]) or {}
                        is_c = r.get("is_correct") == 1
                        icon = "✅" if is_c else "❌"
                        st.markdown(f"**{i}. {itm['stem']}**  {icon}")
                        st.write(f"정답: {itm['answer']}")
                        st.info(itm.get("explanation", "(해설 준비중)"))

# =============== 지표 함수 ===============
def mastery_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["subtopic","acc","speed_idx","recency","S_k","n"])

    grp = df.groupby("subtopic").agg(
        acc=("is_correct","mean"),
        n=("is_correct","count"),
        med_rt=("response_time","median")
    ).reset_index()

    med = grp["med_rt"].fillna(grp["med_rt"].median())
    if med.nunique() == 1:
        speed_idx = pd.Series([0.5]*len(grp), index=grp.index)
    else:
        speed_idx = (med.max() - med) / (med.max() - med.min())
    grp["speed_idx"] = speed_idx

    df["ts_dt"] = pd.to_datetime(df["ts"], errors="coerce")
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=7)
    recent_mask = df["ts_dt"] >= cutoff
    rec = df[recent_mask].groupby("subtopic")["is_correct"].count()
    total = df.groupby("subtopic")["is_correct"].count()
    recency = (rec / total).reindex(grp["subtopic"]).fillna(0)
    grp["recency"] = recency.values

    grp["S_k"] = 0.7*grp["acc"] + 0.2*grp["speed_idx"] + 0.1*grp["recency"]
    return grp

# =============== 탭2: 결과/보강 ===============
with TABS[1]:
    st.subheader("결과 리포트 & 보강 제안")
    try:
        all_resp = pd.read_csv(RESPONSES_CSV)
    except Exception:
        all_resp = pd.DataFrame(columns=["user_id","subtopic","is_correct","response_time","area","level","ts","error_tag"])    

    if user["user_name"]:
        mine = all_resp[all_resp["user_id"]==user["user_id"]].copy()
    else:
        st.info("좌측에서 이름을 입력하고 퀴즈를 먼저 진행하세요.")
        mine = pd.DataFrame()

    if mine.empty:
        st.warning("아직 기록이 없습니다.")
    else:
        col1, col2 = st.columns([1,1])
        with col1:
            st.markdown("**하위개념별 지표**")
            g = mastery_scores(mine)
            st.dataframe(g[["subtopic","acc","speed_idx","recency","S_k","n"]].round(3))
            bar = alt.Chart(g).mark_bar().encode(
                y=alt.Y('subtopic:N', sort='-x', title='하위개념'),
                x=alt.X('S_k:Q', scale=alt.Scale(domain=[0,1]), title='숙달스코어(0~1)'),
                tooltip=['subtopic','acc','speed_idx','recency','S_k']
            ).properties(height=300)
            st.altair_chart(bar, use_container_width=True)
        with col2:
            st.markdown("**하위개념 × 정답률 히트맵**")
            heat = alt.Chart(g).mark_rect().encode(
                y=alt.Y('subtopic:N', sort='-x'),
                x=alt.X('acc:Q', bin=alt.Bin(maxbins=10), title='정답률 구간'),
                color=alt.Color('count():Q', title='빈도'),
                tooltip=['subtopic','acc','n']
            ).properties(height=300)
            st.altair_chart(heat, use_container_width=True)

        weak = g[g["S_k"] < 0.75].sort_values("S_k")
        st.markdown("### 취약영역 제안")
        if weak.empty:
            st.success("모든 개념이 양호합니다. 🎉 재평가로 완전학습을 확인하세요!")
        else:
            for _, row in weak.iterrows():
                sub = row['subtopic']
                st.warning(f"**{sub}** · 숙달 {row['S_k']:.2f} — 보강 권장")
                sample = items_df[items_df["subtopic"]==sub].head(2)
                for __, it in sample.iterrows():
                    with st.expander(f"보강 카드: {it['item_id']} — {it['stem'][:50]}..."):
                        st.write("**핵심 개념 요약**")
                        st.info(it.get("explanation","(해설 준비중)"))
                        st.write("**연습 문항 제안**: 동일/유사 유형 2~3개를 추가하세요.")

        st.markdown("### 오류 유형 요약")
        err = mine.dropna(subset=["error_tag"])  
        if err.empty:
            st.caption("기록된 오류 유형이 없습니다. 다음 풀이에서 스스로 태깅해 보세요.")
        else:
            cnt = err.groupby("error_tag").size().reset_index(name="count").sort_values("count", ascending=False)
            st.dataframe(cnt)

# =============== 탭3: 재평가 ===============
with TABS[2]:
    st.subheader("재평가 (취약영역만)")
    if user["user_name"]:
        try:
            all_resp = pd.read_csv(RESPONSES_CSV)
        except Exception:
            all_resp = pd.DataFrame()
        mine = all_resp[all_resp["user_id"]==user["user_id"]].copy()
        g = mastery_scores(mine)
        weak_list = g[g["S_k"] < 0.75]["subtopic"].tolist()
        if not weak_list:
            st.success("취약영역이 없습니다. 최근 기록 기준으로 통과입니다!")
        else:
            sub_sel = st.multiselect("재평가할 하위개념 선택", weak_list, default=weak_list[:1])
            n_items = st.number_input("문항 수", 3, 12, value=6)
            start = st.button("재평가 시작")
            if start and sub_sel:
                subset = items_df[items_df["subtopic"].isin(sub_sel)]
                pool = subset.sample(min(n_items, len(subset))) if not subset.empty else pd.DataFrame()
                if pool.empty:
                    st.error("해당 개념 문항이 부족합니다. 관리자에게 문의하세요.")
                else:
                    st.info("문항을 순서대로 풉니다. 제출 시 즉시 판정합니다.")
                    attempt_id = str(uuid.uuid4())
                    for i, (_, it) in enumerate(pool.iterrows(), start=1):
                        st.markdown(f"#### R{i}. {it['stem']}")
                        if isinstance(it.get("choices"), list) and it["choices"]:
                            a = st.radio("정답 선택", it["choices"], index=None, key=f"re_choice_{attempt_id}_{i}")
                        else:
                            a = st.text_input("답 입력", key=f"re_text_{attempt_id}_{i}")
                        btn = st.button("제출", key=f"re_submit_{attempt_id}_{i}")
                        if btn:
                            ans = str(a).strip()
                            gold = str(it["answer"]).strip()
                            is_correct = ans.replace(" ", "").lower() == gold.replace(" ", "").lower()
                            row = {
                                "ts": _now_str(),
                                "user_id": user["user_id"],
                                "user_name": user["user_name"],
                                "role": user["role"],
                                "area": it["area"],
                                "subtopic": it["subtopic"],
                                "item_id": it["item_id"],
                                "is_correct": int(is_correct),
                                "response": ans,
                                "response_time": None,
                                "error_tag": None,
                                "level": it["level"],
                                "attempt_id": attempt_id,
                            }
                            try:
                                old = pd.read_csv(RESPONSES_CSV)
                                old.loc[len(old)] = row
                                old.to_csv(RESPONSES_CSV, index=False, encoding="utf-8-sig")
                            except Exception as e:
                                st.warning(f"저장 오류: {e}")

                            if is_correct:
                                st.success("정답! 👍")
                            else:
                                st.error("오답")
                                with st.expander("해설"):
                                    st.write(it.get("explanation","(해설 준비중)"))
    else:
        st.info("좌측에서 이름/학년을 입력하세요.")

# =============== 탭4: 교사 대시보드(항상 접근가능) ===============
with TABS[3]:
    st.subheader("교사 대시보드")
    try:
        df = pd.read_csv(RESPONSES_CSV)
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        st.warning("응답 데이터가 없습니다.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("총 응답 수", len(df))
        with c2:
            st.metric("전체 정답률", f"{100*df['is_correct'].mean():.1f}%")
        with c3:
            st.metric("최근 기록 시각", df['ts'].iloc[-1] if 'ts' in df.columns and not df.empty else '-')

        stu_list = sorted(df['user_name'].dropna().unique().tolist())
        stu_sel = st.multiselect("학생 선택(빈칸=전체)", stu_list)
        if stu_sel:
            df = df[df['user_name'].isin(stu_sel)]

        topn = st.slider("오답 상위 N", 3, 15, 7)
        wrong = df[df['is_correct']==0]
        if wrong.empty:
            st.success("오답 데이터가 거의 없습니다. 👍")
        else:
            top = wrong.groupby(['area','subtopic']).size().reset_index(name='cnt').sort_values('cnt', ascending=False).head(topn)
            st.markdown("**오답 빈발 개념 TOP N**")
            st.dataframe(top)

        st.markdown("### 학생×하위개념 정답률 히트맵")
        piv = df.pivot_table(index='user_name', columns='subtopic', values='is_correct', aggfunc='mean')
        piv = piv.reset_index().melt('user_name', var_name='subtopic', value_name='acc')
        heat = alt.Chart(piv).mark_rect().encode(
            y=alt.Y('user_name:N', sort='-x', title='학생'),
            x=alt.X('subtopic:N', title='하위개념'),
            color=alt.Color('acc:Q', scale=alt.Scale(domain=[0,1]), title='정답률'),
            tooltip=['user_name','subtopic',alt.Tooltip('acc:Q', format='.2f')]
        ).properties(height=300)
        st.altair_chart(heat, use_container_width=True)

        st.download_button(
            "응답 원시데이터 CSV 다운로드",
            data=df.to_csv(index=False, encoding='utf-8-sig'),
            file_name=f"responses_export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime='text/csv'
        )

# =============== 푸터 ===============
st.divider()
st.caption("ⓒ 개념 마스터(MVP) — Streamlit 샘플. 실제 운영 시 계정/권한, 보안, 난이도 보정, 대규모 문항은행 등을 확장하세요.")
