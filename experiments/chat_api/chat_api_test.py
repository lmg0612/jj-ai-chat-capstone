# ✅ 시스템 프롬프트 기반 1차 + 2차 필터링 구조 (system + 사용자 프롬프트 통합)
import requests
import pandas as pd
import re
import json
import os

# === 경로 설정 ===
BASE_DIR = "prompt"
SYSTEM_PROMPT_PATH = os.path.join(BASE_DIR, "system_prompt.txt")
PROMPT_INPUT_PATH = os.path.join(BASE_DIR, "prompt_input.txt")
CSV_PATH = r"C:\Users\dlakr\OneDrive\바탕 화면\chat_api\일반_카테고리_점수환산(6-4-0)_test.csv"
CATEGORY_KEYWORDS_PATH = os.path.join(BASE_DIR, "category_keywords.json")
OLLAMA_URL = "http://localhost:11434/api/chat"

# === 데이터 로딩 ===
with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
    system_prompt = f.read()

try:
    with open(CATEGORY_KEYWORDS_PATH, "r", encoding="utf-8") as f:
        category_keywords = json.load(f)
except Exception as e:
    raise RuntimeError(f"[카테고리 키워드 로딩 실패] {e}")

try:
    df_notice = pd.read_csv(CSV_PATH)
except Exception as e:
    raise RuntimeError(f"[공지 CSV 로딩 실패] {e}")

# === 언어 감지 ===
def detect_language(text):
    if re.search(r'[\uac00-\ud7a3]', text): return 'ko'
    elif re.search(r'[\u3040-\u30ff]', text): return 'ja'
    elif re.search(r'[\u4e00-\u9fff]', text): return 'zh'
    elif re.search(r'[ăâđêôơư]', text.lower()): return 'vi'
    else: return 'en'

# === 1차 필터: 키워드 기반 유사도 계산 후 상위 공지 추출 ===
def filter_top_notices(user_input, top_n=20):
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

# === 프롬프트 저장: system 프롬프트 + 사용자 질문 + 공지 목록 포함 ===
def save_prompt_file(question, top_df):
    with open(PROMPT_INPUT_PATH, "w", encoding="utf-8") as f:
        f.write(system_prompt.strip())
        f.write(f"\n\n사용자 질문:\n{question}\n\n")
        f.write("다음은 관련 공지사항 목록입니다. 가장 관련 있는 5개만 선택해 요약해 주세요.\n\n")
        for i, row in top_df.iterrows():
            f.write(f"{i+1}. 제목: {row['공지 제목']}\n")
            f.write(f"내용: {str(row.get('본문 내용', '')).strip()}\n")
            f.write(f"OCR: {str(row.get('이미지 텍스트', '')).strip()}\n")
            f.write(f"링크: {row.get('링크', '')}\n\n")

# === Gemma 호출 ===
def query_gemma_from_file():
    with open(PROMPT_INPUT_PATH, "r", encoding="utf-8") as f:
        prompt = f.read()

    response = requests.post(OLLAMA_URL, json={
        "model": "gemma3:4b",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": True
    }, stream=True)

    print("🤖 응답:")
    full = ""
    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                data = line.decode("utf-8")
                if '"content":"' in data:
                    content = data.split('"content":"')[1].split('"')[0].replace('\\n', '\n')
                    print(content, end="", flush=True)
                    full += content
    else:
        print("❌ 오류:", response.text)
    return full

# === 메인 실행 ===
print("💬 전주대 챗봇 공지 요약 시스템 (2차 필터 포함)\n")
while True:
    question = input("👤 질문: ")
    if question.lower() in ['exit', '종료', 'quit']:
        print("👋 종료합니다.")
        break
    lang = detect_language(question)
    top_df = filter_top_notices(question, top_n=20)
    if top_df.empty:
        print("❗ 관련 공지 없음")
        continue
    save_prompt_file(question, top_df)
    query_gemma_from_file()
    print()
