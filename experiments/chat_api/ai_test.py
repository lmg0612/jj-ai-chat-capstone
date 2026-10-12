import requests
import pandas as pd
import re
import json
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# === 경로 설정 ===
BASE_DIR = "prompt"
DEFAULT_PROMPT_PATH = os.path.join(BASE_DIR, "system_prompt_default.txt")
SUMMARY_PROMPT_PATH = os.path.join(BASE_DIR, "system_prompt_notice_summary.txt")
CSV_PATH = r"C:\Users\dlakr\OneDrive\바탕 화면\chat_api\일반_카테고리_점수환산(6-4-0)_test.csv"
CATEGORY_KEYWORDS_PATH = os.path.join(BASE_DIR, "category_keywords.json")
OLLAMA_URL = "http://localhost:11434/api/chat"

# === 데이터 로딩 ===
with open(DEFAULT_PROMPT_PATH, "r", encoding="utf-8") as f:
    system_prompt_default = f.read()
with open(SUMMARY_PROMPT_PATH, "r", encoding="utf-8") as f:
    system_prompt_summary = f.read()
with open(CATEGORY_KEYWORDS_PATH, "r", encoding="utf-8") as f:
    category_keywords = json.load(f)

df_notice = pd.read_csv(CSV_PATH)
encoder = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# === 언어 감지 ===
def detect_language(text):
    if re.search(r'[\uac00-\ud7a3]', text): return 'ko'
    elif re.search(r'[\u3040-\u30ff]', text): return 'ja'
    elif re.search(r'[\u4e00-\u9fff]', text): return 'zh'
    elif re.search(r'[ăâđêôơư]', text.lower()): return 'vi'
    else: return 'en'

# === 공지 키워드 포함 여부
def is_notice_question(q):
    return any(kw in q for kws in category_keywords.values() for kw in kws)

# === 점수 기반 필터
def filter_top_notices_keyword(user_input, top_n=20):
    scores = []
    for idx, row in df_notice.iterrows():
        text = " ".join([
            str(row.get("공지 제목", "")),
            str(row.get("본문 내용", "")),
            str(row.get("이미지 텍스트", ""))
        ]).lower()
        score = sum(text.count(kw.lower()) for cat in category_keywords.values() for kw in cat)
        scores.append((idx, score))
    sorted_indices = sorted(scores, key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, score in sorted_indices if score > 0][:top_n]
    return df_notice.loc[top_indices].reset_index(drop=True)

# === 의미 기반 필터
def filter_top_notices_semantic(question, candidate_df, top_n=5):
    question_vec = encoder.encode([question])
    candidate_texts = [
        " ".join([
            str(row.get("공지 제목", "")),
            str(row.get("본문 내용", "")),
            str(row.get("이미지 텍스트", ""))
        ]) for _, row in candidate_df.iterrows()
    ]
    candidate_vecs = encoder.encode(candidate_texts)
    similarities = cosine_similarity(question_vec, candidate_vecs)[0]
    bonus = [sum(1 for kw in question.lower().split() if kw in text.lower()) for text in candidate_texts]
    final_scores = [sim + 0.15 * b for sim, b in zip(similarities, bonus)]
    top_indices = sorted(range(len(final_scores)), key=lambda i: final_scores[i], reverse=True)[:top_n]
    return candidate_df.iloc[top_indices].drop_duplicates(subset=["공지 제목", "링크"]).reset_index(drop=True)

# === Gemma 호출
def query_gemma(system_prompt, user_prompt):
    response = requests.post(OLLAMA_URL, json={
        "model": "gemma3:4b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": True
    }, stream=True)

    print("🤖 응답:")
    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                data = line.decode("utf-8")
                if '\"content\":\"' in data:
                    content = (
                        data.split('\"content\":\"')[1].split('\"')[0]
                        .replace('\\n', '\n')
                        .replace('\\u0026', '&')
                        .replace('\u0026', '&')
                    )
                    print(content, end="", flush=True)
    else:
        print("❌ 오류:", response.text)

# === 공지 질문 처리
def handle_notice_summary(question):
    top20 = filter_top_notices_keyword(question, top_n=20)
    top5 = filter_top_notices_semantic(question, top20, top_n=5)

    prompt = f"사용자 질문:\n{question}\n\n공지 목록:\n"
    for _, row in top5.iterrows():
        prompt += f"[{row['공지 제목']}]: {row['링크']}\n"

    query_gemma(system_prompt_summary, prompt)

# === 일반 질문 처리
def handle_general(question):
    query_gemma(system_prompt_default, question)

# === 실행 루프
if __name__ == "__main__":
    print("💬 전주대 챗봇 JJ (프롬프트 기반 응답 시스템)\n")
    while True:
        question = input("👤 질문: ")
        if question.lower() in ['exit', '종료', 'quit']:
            print("👋 종료합니다.")
            break

        print(f"🔹 Step 1: 질문 수신됨 → {question}")
        lang = detect_language(question)
        print(f"🔹 Step 2: 감지된 언어: {lang}")

        if is_notice_question(question):
            print("🔹 Step 3: 공지 키워드 포함 → 공지 요약 처리")
            handle_notice_summary(question)
        else:
            print("🔹 Step 3: 일반 질문 → 기본 프롬프트 처리")
            handle_general(question)

        print()
