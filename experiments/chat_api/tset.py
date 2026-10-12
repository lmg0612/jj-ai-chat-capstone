import requests
import pandas as pd
import re
import json
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# === 경로 설정 ===
BASE_DIR = "prompt"
PROMPT_PATH = os.path.join(BASE_DIR, "system_prompt_default.txt")
CSV_PATH = r"C:\Users\dlakr\OneDrive\바탕 화면\chat_api\일반_카테고리_점수환산(6-4-0)_test.csv"
CATEGORY_PATH = os.path.join(BASE_DIR, "category_keywords.json")
OLLAMA_URL = "http://localhost:11434/api/chat"

# === 프롬프트 및 데이터 로딩 ===
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()
with open(CATEGORY_PATH, "r", encoding="utf-8") as f:
    category_keywords = json.load(f)
df_notice = pd.read_csv(CSV_PATH, encoding="utf-8")

# === 언어 감지 ===
def detect_language(text):
    if re.search(r'[\uac00-\ud7a3]', text): return 'ko'
    elif re.search(r'[\u3040-\u30ff]', text): return 'ja'
    elif re.search(r'[\u4e00-\u9fff]', text): return 'zh'
    elif re.search(r'[ăâđêôơư]', text.lower()): return 'vi'
    else: return 'en'

# === 인사 또는 정체성 질문 감지 ===
def is_greeting_or_identity(q):
    patterns = ["안녕", "누구", "역할", "정체", "뭐해", "하는일", "챗봇", "너는"]
    return any(p in q.lower() for p in patterns)

# === 세부 조건 질문 여부 판단 ===
def is_detailed_query(q):
    base = ["전공", "학과", "AI", "소프트웨어", "프로그래밍", "컴퓨터", "공학"]
    qualifier = ["관련", "대상", "조건", "참여", "지원", "누가", "어떤"]
    return any(b in q for b in base) and any(qf in q for qf in qualifier)

# === 카테고리 점수 계산 ===
def get_top_category(question):
    scores = {cat: 0 for cat in category_keywords}
    for cat, kws in category_keywords.items():
        scores[cat] = sum(kw in question for kw in kws)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores[0][0] if sorted_scores[0][1] > 0 else None

# === 단순 공지 필터링 ===
def keyword_filter(df, category):
    top5 = df[df[category] > 0].sort_values(by=category, ascending=False).head(5)
    result = f"[{category}]에 대한 공지입니다.\n"
    for _, row in top5.iterrows():
        result += f"[{row['공지 제목']}]: {row['링크']}\n"
    return result

# === SBERT 임베딩 기반 필터링 ===
encoder = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

def semantic_filter(question, df, topn=5):
    q_vec = encoder.encode([question])

    scores = []
    for _, row in df.iterrows():
        t = str(row.get("공지 제목", ""))
        b = str(row.get("본문 내용", ""))
        o = str(row.get("이미지 텍스트", ""))

        t_sim = cosine_similarity(encoder.encode([t]), q_vec)[0][0]
        b_sim = cosine_similarity(encoder.encode([b]), q_vec)[0][0] if b.strip() else 0
        o_sim = cosine_similarity(encoder.encode([o]), q_vec)[0][0] if o.strip() else 0

        score = 0.5 * t_sim + 0.3 * b_sim + 0.2 * o_sim
        scores.append(score)

    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topn]
    return df.iloc[top_idx].drop_duplicates(subset=["공지 제목", "링크"]).reset_index(drop=True)

# === 프롬프트 + Gemma 호출 ===
def ask_llm(prompt_text):
    response = requests.post(OLLAMA_URL, json={
        "model": "gemma3:4b",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text}
        ],
        "stream": True
    }, stream=True)

    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                data = line.decode("utf-8")
                if '"content":"' in data:
                    content = data.split('"content":"')[1].split('"')[0]
                    print(content.replace('\\n', '\n').replace('\\u0026', '&'))
    else:
        print("❌ 오류:", response.text)

# === 메인 실행 ===
print("💬 전주대 챗봇 JJ (프롬프트 기반 응답 시스템)\n")
while True:
    q = input("👤 질문: ")
    if q.lower() in ["exit", "종료", "quit"]:
        print("👋 종료합니다.")
        break

    print(f"🔹 Step 1: 질문 수신됨 → {q}")
    lang = detect_language(q)
    print(f"🔹 Step 2: 감지된 언어: {lang}")

    if is_greeting_or_identity(q):
        print("🔹 Step 3: 인사 또는 정체성 질문 감지 → 인사 응답")
        print("안녕하세요! 저는 전주대학교 학생들을 위한 캠퍼스 도우미 챗봇 'JJ'입니다.\n")
        continue

    category = get_top_category(q)
    print(f"🔹 Step 3: 상위 카테고리: {category}")

    if is_detailed_query(q):
        print("🔹 Step 4: 세부 질문 감지 → 2단계 필터링")
        top_50 = df_notice[df_notice[category] > 0].sort_values(by=category, ascending=False).head(50)
        refined = semantic_filter(q, top_50, topn=5)

        if refined.empty:
            print("❗ 관련 공지 없음")
        else:
            print(f"[{category}]에 대한 공지입니다.")
            for _, row in refined.iterrows():
                print(f"[{row['공지 제목']}]: {row['링크']}")
    else:
        print("🔹 Step 4: 단순 공지 요청 → 상위 5개 즉시 출력")
        print(keyword_filter(df_notice, category))
