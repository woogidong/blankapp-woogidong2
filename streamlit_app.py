# app.py — Math Concept Mastery (LaTeX + Robust CSV choices parsing)
# ---------------------------------------------------------------
# 변경점 요약
# - choices가 비어 있거나 "None" 문자열이어도 주관식 입력창이 뜨도록 수정
# - LaTeX 렌더링
# - 퀴즈 종료 시 전체 해설
# - 교사 대시보드 항상 열람 가능
# - 업로드 CSV 스키마/파싱 안전화

import os
import json
import time
import uuid
from datetime import datetime
from typing import Optional, List, Any

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# ================== App Config ==================
st.set_page_config(page_title="수학 개념 진단 클리닉", layout="wide")

# ====== Custom button / UI styling ======
st.markdown(
    """
    <style>
    /* Primary-looking buttons */
    .stButton>button {
        background: linear-gradient(90deg,#4f9eed,#2b7bd3);
        color: white;
        border: none;
        padding: 8px 14px;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(43,123,211,0.25);
    }
    .stButton>button:focus { outline: none; }
    .stButton>button:hover { transform: translateY(-1px); }

    /* Secondary smaller controls (toggles/selects) spacing */
    .stSelectbox, .stNumberInput, .stMultiSelect { margin-bottom: 8px; }

    /* Emphasize submit button */
    button[kind="primary"] {
        background: linear-gradient(90deg,#f59e0b,#f97316) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

APP_TITLE = "수학 개념 진단 클리닉"
DATA_DIR = "data"
RESPONSES_CSV = os.path.join(DATA_DIR, "responses.csv")
USERS_CSV = os.path.join(DATA_DIR, "users.csv")
os.makedirs(DATA_DIR, exist_ok=True)
TERMS_JSON = os.path.join(DATA_DIR, "math_terms.json")


def load_terms() -> dict:
    try:
        with open(TERMS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "함수": "함수는 각 입력값에 대해 정확히 하나의 출력값이 대응하는 규칙 또는 관계입니다.",
            "미분": "미분은 함수의 순간 변화율을 구하는 연산입니다. 도함수 f'(x)는 x에서의 기울기를 나타냅니다.",
        }

# ================== Utils & Schema ==================
def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

REQUIRED_USER_COLS = ["user_id", "user_name", "role", "grade", "age", "created_at"]
REQUIRED_ITEM_COLS = {
    "item_id","area","subtopic","level","time_hint",
    "stem","choices","answer","explanation","error_tags"
}

def _empty_users_df() -> pd.DataFrame:
    return pd.DataFrame(columns=REQUIRED_USER_COLS)

def load_users_df() -> pd.DataFrame:
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
    return df[REQUIRED_USER_COLS]

# initialize data files
if not os.path.exists(RESPONSES_CSV):
    pd.DataFrame(columns=[
        "ts","user_id","user_name","role","area","subtopic","item_id","is_correct",
        "response","response_time","error_tag","level","attempt_id"
    ]).to_csv(RESPONSES_CSV, index=False, encoding="utf-8-sig")

if not os.path.exists(USERS_CSV):
    _empty_users_df().to_csv(USERS_CSV, index=False, encoding="utf-8-sig")

# ================== Session State ==================
if "user" not in st.session_state:
    st.session_state.user = {"user_id": None, "user_name": None, "role": "학생", "grade": None, "age": None}
if "quiz" not in st.session_state:
    st.session_state.quiz = {
        "pool": [], "current_idx": 0, "start_ts": None, "attempt_id": None,
        "area": None, "levels": ["L1","L2","L3"], "size": 8
    }
if "responses" not in st.session_state:
    st.session_state.responses = []

# ================== LaTeX Helper ==================
def render_latex_or_text(s: Optional[str], *, label: Optional[str]=None) -> None:
    if s is None:
        return
    s = str(s)
    math_triggers = ["\\frac", "\\sqrt", "^", "_", "\\sum", "\\int", "\\lim", "\\rightarrow", "\\to", "\\pm", "\\ge", "\\le", "\\cdot", "\\times"]

    # 레이블 출력
    if label:
        st.markdown(f"**{label}**")

    # 이미 $...$ 가 포함되어 있으면, 상황에 맞게 렌더링
    if "$" in s:
        stripped = s.strip()
        # 순수 수식(전체가 하나의 $...$로 감싸인 경우)은 st.latex로 블록 렌더링
        if stripped.startswith("$") and stripped.endswith("$") and stripped.count("$") == 2:
            inner = stripped.strip("$")
            try:
                st.latex(inner)
                return
            except Exception:
                # 실패하면 마크다운으로 폴백
                st.markdown(stripped)
                return
        # 텍스트와 수식이 섞여 있거나 여러 수식이 있는 경우는 마크다운으로 인라인 렌더링
        st.markdown(s)
        return

    # $가 없지만 LaTeX 트리거가 포함되어 있으면 인라인 수식 형태로 마크다운에 감싸서 렌더링
    if any(t in s for t in math_triggers):
        try:
            st.markdown(f"${s}$")
            return
        except Exception:
            pass

    # 기본 텍스트
    st.write(s)

# ================== Robust JSON-ish parser ==================
def parse_jsonish_list(x: Any):
    """
    choices / error_tags에 쓰는 안전 파서.
    - 빈칸/NaN/"None"(대소문자 무관) → None (주관식으로 처리)
    - JSON 배열 문자열 → list로 파싱 (스마트따옴표/홑따옴표 보정 재시도)
    - 그 외 → 원문 유지
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, list):
        return x
    if not isinstance(x, str):
        return x

    s = x.strip()
    if s == "" or s.lower() == "none":
        return None

    if s.startswith("["):
        try:
            return json.loads(s)
        except Exception:
            s2 = (s.replace("“","\"").replace("”","\"")
                    .replace("’","'").replace("′","'")
                    .replace("，",","))
            # 홑따옴표만 있고 쌍따옴표가 없다면 교체
            if ("\"" not in s2) and ("'" in s2):
                s2 = s2.replace("'", "\"")
            try:
                return json.loads(s2)
            except Exception:
                # 파싱 실패 시 원문 문자열 유지 (디버깅 위해)
                return s
    return s

# ================== Seed Items ==================
def load_seed_items() -> pd.DataFrame:
    seed = [
        {"item_id":"ALG-001","area":"대수","subtopic":"다항식 전개","level":"L1","time_hint":30,
         "stem":"(x+2)(x-3)를 전개하시오.","choices":None,"answer":"$x^2 - x - 6$",
         "explanation":"$x(x-3)+2(x-3)=x^2-3x+2x-6=x^2-x-6$","error_tags":["절차오류","계산실수"]},
        {"item_id":"ALG-002","area":"대수","subtopic":"인수분해","level":"L1","time_hint":35,
         "stem":"$x^2-5x+6$ 을 인수분해하시오.","choices":None,"answer":"$(x-2)(x-3)$",
         "explanation":"곱 6, 합 5 → 2와 3","error_tags":["개념미이해"]},
        {"item_id":"FUN-004","area":"함수","subtopic":"최대최소","level":"L3","time_hint":75,
         "stem":"함수 $f(x)=x^2-4x+5$ 의 최솟값은?","choices":None,"answer":"$1$",
         "explanation":"$(x-2)^2+1$ → 최솟값 1","error_tags":["개념미이해"]},
        {"item_id":"GEO-002","area":"기하","subtopic":"피타고라스","level":"L1","time_hint":45,
         "stem":"직각삼각형에서 빗변이 13, 한 변이 5일 때 다른 변은?","choices":None,"answer":"$12$",
         "explanation":"$13^2-5^2=169-25=144 → 12$","error_tags":["계산실수"]},
        {"item_id":"STA-004","area":"확률과 통계","subtopic":"표준편차","level":"L2","time_hint":70,
         "stem":"데이터 $1,3,5$ 의 표준편차(모표준편차)를 구하시오.","choices":None,"answer":"$\\approx 1.632$",
         "explanation":"평균 3, 분산 $8/3$ → $\\sigma=\\sqrt{8/3}\\approx1.632$","error_tags":["계산실수"]},
        {"item_id":"FUN-002","area":"함수","subtopic":"일차함수","level":"L2","time_hint":60,
         "stem":"$y=3x-2$ 의 기울기는?","choices":["$-2$","$0$","$3$","$\\tfrac{2}{3}$"],"answer":"$3$",
         "explanation":"$y=mx+b$ 에서 $m=3$","error_tags":["개념미이해"]},
    ]
    return pd.DataFrame(seed)

# ================== Items Load (Upload + Sanitize) ==================
def sanitize_and_parse_items(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 누락 열 보강
    missing = REQUIRED_ITEM_COLS - set(df.columns)
    for c in missing:
        df[c] = np.nan
    # 컬럼 순서 표준화
    df = df[[
        "item_id","area","subtopic","level","time_hint",
        "stem","choices","answer","explanation","error_tags"
    ]]
    # 이상 행 제거
    df = df[
        df["item_id"].astype(str).str.strip().ne("") &
        df["area"].astype(str).str.strip().ne("") &
        df["stem"].astype(str).str.strip().ne("")
    ].reset_index(drop=True)
    # 안전 파싱
    for col in ["choices","error_tags"]:
        df[col] = df[col].apply(parse_jsonish_list)
    return df

def load_items_from_upload(uploaded: Optional[pd.DataFrame]) -> pd.DataFrame:
    base = load_seed_items()
    if uploaded is None:
        return sanitize_and_parse_items(base)
    try:
        parsed = sanitize_and_parse_items(uploaded)
        # 비어 있으면 시드로 대체
        if parsed.empty:
            return sanitize_and_parse_items(base)
        return parsed
    except Exception:
        return sanitize_and_parse_items(base)

# ================== Sidebar Login ==================
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
                "user_id": uid, "user_name": user_name, "role": role,
                "grade": grade_val, "age": age_val, "created_at": _now_str(),
            }
            users_df = pd.concat([users_df, pd.DataFrame([new_row])], ignore_index=True)
            users_df.to_csv(USERS_CSV, index=False, encoding="utf-8-sig")
            st.success(f"환영합니다, {user_name} (학생)")
    # 학생 정보 초기화 버튼
    if st.button("학생 정보 초기화", key="reset_user"):
        st.session_state.user = {"user_id": None, "user_name": None, "role": "학생", "grade": None, "age": None}
        st.success("학생 정보가 초기화되었습니다.")

# ================== Header ==================
st.title(APP_TITLE)
user = st.session_state.user
if user["user_name"]:
    extra = []
    if user.get("grade"): extra.append(f"학년: {user['grade']}")
    if user.get("age") is not None: extra.append(f"나이: {user['age']}")
    st.caption(" · ".join(filter(None, [f"접속: {user['user_name']}", "역할: 학생"] + extra)))

# ================== Tabs ==================
TABS = st.tabs(["퀴즈", "결과/보강", "교사 대시보드", "문항 업로드", "용어사전"])

# ================== Items Upload Tab ==================
with TABS[3]:
    st.subheader("문항 업로드 (CSV)")
    st.write("필수 컬럼: item_id, area, subtopic, level, time_hint, stem, choices, answer, explanation, error_tags")
    st.write("- **주관식**: choices를 공란/`None`(문자열) → 자동으로 입력창 표시")
    st.write("- **객관식**: choices를 JSON 배열로 (예: `[\"$1$\",\"$2$\",\"$3$\",\"$4$\"]` )")
    uploaded_file = st.file_uploader("CSV 업로드", type=["csv"])
    uploaded_df = None
    if uploaded_file:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.success(f"업로드 성공: {uploaded_df.shape}")
            st.dataframe(uploaded_df.head())
        except Exception as e:
            st.error(f"업로드 실패: {e}")

items_df = load_items_from_upload(uploaded_df)

# ================== Quiz Utilities ==================
def build_quiz_pool(df: pd.DataFrame, area: Optional[str], levels: List[str], size: int) -> List[dict]:
    """
    Build a quiz pool. If area is None, sample across all areas (only filter by levels).
    """
    if area is None:
        subset = df[df["level"].isin(levels)].copy()
    else:
        subset = df[(df["area"] == area) & (df["level"].isin(levels))].copy()
    if subset.empty:
        return []
    pool = subset.sample(n=min(size, len(subset)), replace=False, random_state=None)
    return pool.to_dict(orient="records")

# ================== Tab 1: Quiz ==================
with TABS[0]:
    st.subheader("영역별 퀴즈")
    cols = st.columns([1,1,1,1,1])
    with cols[0]:
        # 영역 선택 옵션 제거: 문제는 무작위로 제시됩니다.
        st.markdown("**영역: 무작위**")
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
            # area is None so pool is sampled across all areas randomly
            "pool": build_quiz_pool(items_df, None, levels, size),
            "current_idx": 0,
            "start_ts": time.time() if include_timer else None,
            "attempt_id": str(uuid.uuid4()),
            "area": None,
            "levels": levels,
            "size": size,
        })
        st.success(f"퀴즈 생성: {len(st.session_state.quiz['pool'])}문항")

    quiz = st.session_state.quiz
    if quiz["pool"]:
        q = quiz["pool"][quiz["current_idx"]]
        st.markdown(f"#### Q{quiz['current_idx']+1}.")
        render_latex_or_text(q.get("stem"), label="문제")

        # -------- choices 처리: LaTeX 보기를 보여주고 라디오로 선택하도록 --------
        choices = q.get("choices")
        sel_label = None
        if isinstance(choices, list) and len(choices) > 0:
            # 객관식: 위에는 LaTeX 렌더된 보기, 아래에는 A/B/C 라디오로 선택
            letters = [chr(ord('A') + i) for i in range(len(choices))]
            for i, ch in enumerate(choices):
                # 각 선택지를 LaTeX/텍스트 혼합으로 렌더
                render_latex_or_text(f"{letters[i]}. {str(ch)}")
            sel_label = st.radio("정답 선택 (위의 선택지를 확인하세요)", letters, index=None, key=f"choiceidx_{quiz['attempt_id']}_{quiz['current_idx']}")
            # 선택한 레이블을 실제 문자열로 매핑 (선택이 없으면 None)
            if sel_label in letters:
                user_answer = choices[letters.index(sel_label)]
            else:
                user_answer = ""
        else:
            # 주관식 (choices가 None, 빈칸, "None" 문자열이었던 경우 모두 여기로)
            user_answer = st.text_input("답 입력 (LaTeX 가능)", key=f"input_{quiz['attempt_id']}_{quiz['current_idx']}")

        # 버튼/입력 배치: 오류 태그는 왼쪽, 제출 버튼은 오른쪽(눈에 띄게)
        c_left, c_right = st.columns([3,1])
        with c_left:
            err_tag = st.selectbox("(선택) 내가 생각하는 오류 유형", ["선택안함","개념미이해","절차오류","계산실수","문제해석","시간관리"])
        with c_right:
            submit = st.button("제출", type="primary", use_container_width=True)

        # 제출 전 검증: 객관식 문항인 경우 반드시 선택을 해야 함
        if submit:
            if isinstance(choices, list) and len(choices) > 0 and not sel_label:
                st.warning("객관식 문항입니다. 답을 선택한 후 제출하세요.")
                st.stop()
            ans_str = str(user_answer).strip()
            gold = str(q["answer"]).strip()
            norm = lambda s: s.replace(" ", "").lower().replace("\\,", "").strip("$")
            is_correct = norm(ans_str) == norm(gold)

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
                "response": ans_str,
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

            # 즉시 정답/오답 피드백 제거: 사용자 요청에 따라 매 풀이마다 결과를 표시하지 않음
            # (결과는 세트 종료 시 전체 해설에서 확인 가능)

            # 다음 문항 or 종료
            if quiz["current_idx"] < len(quiz["pool"]) - 1:
                quiz["current_idx"] += 1
                st.rerun()
            else:
                st.success("퀴즈 종료! 아래에서 전체 해설을 확인하세요.")
                # 축하 효과: 풍선
                try:
                    st.balloons()
                except Exception:
                    # st.balloons()가 실패해도 진행
                    pass
                with st.expander("📚 이번 세트 전체 해설 보기", expanded=True):
                    resp_list = st.session_state.get("responses", [])
                    attempt_id = quiz["attempt_id"]
                    resp_map = {r["item_id"]: r for r in resp_list if r.get("attempt_id") == attempt_id}
                    for i, itm in enumerate(quiz["pool"], start=1):
                        r = resp_map.get(itm["item_id"]) or {}
                        is_c = r.get("is_correct") == 1
                        icon = "✅" if is_c else "❌"
                        st.markdown(f"**{i}.** {icon}")
                        # 하위개념(subtopic) 표기
                        sub = itm.get("subtopic") or "-"
                        st.markdown(f"**하위개념:** {sub}")
                        render_latex_or_text(itm.get("stem"), label="문제")
                        render_latex_or_text(itm.get("answer"), label="정답")
                        render_latex_or_text(itm.get("explanation"), label="해설")

# ================== Metrics Helpers ==================
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
    rec = df[df["ts_dt"] >= cutoff].groupby("subtopic")["is_correct"].count()
    total = df.groupby("subtopic")["is_correct"].count()
    recency = (rec / total).reindex(grp["subtopic"]).fillna(0)
    grp["recency"] = recency.values
    grp["S_k"] = 0.7*grp["acc"] + 0.2*grp["speed_idx"] + 0.1*grp["recency"]
    return grp

# ================== Tab 2: Results / Remediation ==================
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
            st.markdown("**오류 유형 요약**")
            err = mine.dropna(subset=["error_tag"])
            if err.empty:
                st.caption("기록된 오류 유형이 없습니다. 다음 풀이에서 스스로 태깅해 보세요.")
            else:
                cnt = err.groupby("error_tag").size().reset_index(name="count").sort_values("count", ascending=False)
                st.dataframe(cnt)

        # 최근 오답 목록 (subtopic 포함)
        st.markdown("**최근 오답 문항 (최대 20개)**")
        wrong = mine[mine["is_correct"] == 0].sort_values("ts", ascending=False)
        if wrong.empty:
            st.caption("최근 오답이 없습니다. 계속 풀어보세요!")
        else:
            display_cols = [c for c in ["ts", "item_id", "area", "subtopic", "response", "error_tag"] if c in wrong.columns]
            st.dataframe(wrong[display_cols].head(20).reset_index(drop=True))

        # 하위개념 기반 보충학습 추천
        st.markdown("**추천 보충학습(하위개념 기반)**")
        wrong_by_sub = wrong.groupby("subtopic").size().reset_index(name="cnt").sort_values("cnt", ascending=False)
        if wrong_by_sub.empty:
            st.caption("추천할 보충학습 항목이 없습니다.")
        else:
            # 상위 3개 추천
            topn = wrong_by_sub.head(3)
            for _, row in topn.iterrows():
                sub = row["subtopic"] or "-"
                cnt = int(row["cnt"])
                st.markdown(f"- **{sub}** — 오답 {cnt}회")
                # 보충학습 시작 버튼
                if st.button(f"{sub} 보충학습 시작", key=f"remed_{sub}"):
                    # 해당 subtopic에서 문제를 샘플링하여 퀴즈로 연결
                    subset = items_df[items_df["subtopic"] == sub]
                    if subset.empty:
                        st.warning("해당 하위개념의 문항이 없습니다.")
                    else:
                        pool = subset.sample(n=min(5, len(subset)), replace=False).to_dict(orient="records")
                        st.session_state.quiz.update({
                            "pool": pool,
                            "current_idx": 0,
                            "start_ts": time.time() if include_timer else None,
                            "attempt_id": str(uuid.uuid4()),
                            "area": None,
                            "levels": levels,
                            "size": len(pool),
                            "show_results": False,
                        })
                        st.success(f"보충학습 {sub} 세트를 시작합니다 ({len(pool)}문항)")
                        try:
                            st.experimental_rerun()
                        except Exception:
                            # 일부 환경에서 experimental_rerun이 없거나 동작하지 않을 수 있음
                            # 안전하게 현재 실행 중단하여 UI가 재렌더되도록 함
                            try:
                                st.stop()
                            except Exception:
                                pass

# ================== Tab 3: Teacher Dashboard (Always visible) ==================
with TABS[2]:
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

# ================== Tab 4: Math Terms Dictionary ==================
with TABS[4]:
    st.subheader("수학 용어사전")
    terms = load_terms()
    query = st.text_input("찾을 용어를 입력하세요 (부분검색 가능)")
    if query:
        matches = {k: v for k, v in terms.items() if query in k or query in v}
    else:
        matches = {}

    if not matches and query:
        st.info("검색어와 일치하는 용어가 없습니다.")
    elif not query:
        st.markdown("사전에서 용어를 검색해 보세요. (예: 함수, 미분, 적분)")
    else:
        for term, defi in matches.items():
            st.markdown(f"### {term}")
            st.write(defi)

    st.markdown("---")
    st.markdown("### 용어 등록")
    with st.form("add_term_form"):
        new_term = st.text_input("등록할 용어")
        new_def = st.text_area("정의 입력")
        overwrite = st.checkbox("기존 항목이 있으면 덮어쓰기", value=False)
        add_submitted = st.form_submit_button("용어 등록")
        if add_submitted:
            if not new_term or not new_def:
                st.error("용어와 정의를 모두 입력하세요.")
            else:
                current = load_terms()
                if (new_term in current) and (not overwrite):
                    st.warning("이미 존재하는 용어입니다. 덮어쓰려면 '기존 항목이 있으면 덮어쓰기'를 체크하세요.")
                else:
                    current[new_term] = new_def
                    try:
                        with open(TERMS_JSON, "w", encoding="utf-8") as f:
                            json.dump(current, f, ensure_ascii=False, indent=2)
                        st.success(f"용어 '{new_term}' 이(가) 사전에 등록되었습니다.")
                        terms = current
                    except Exception as e:
                        st.error(f"용어 저장 실패: {e}")