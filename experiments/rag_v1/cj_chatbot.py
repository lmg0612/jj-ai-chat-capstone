import aiohttp
import requests
import pandas as pd
import re
import json
from functools import reduce
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from current_datetime import get_current_datetime_string
import numpy as np

# ====== 🔹 설정 ======
OLLAMA_URL = "http://localhost:11434/api/chat"
NOTICE_CSV_PATH = "/Users/shin/projects/CJ_AI_test/일반_카테고리_점수환산(6-4-0).csv"
SYSTEM_PROMPT_PATH = "/Users/shin/projects/CJ_AI_test/system_prompt.txt"

# ====== 🔹 시스템 프롬프트 로드 (다국어 대응 포함) ======
with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
    base_prompt = f.read()

MULTILANG_PROMPTS = {
    "ko": base_prompt + "\n\n👉 모든 응답은 반드시 한국어로 작성해 주세요.",
    "en": base_prompt + "\n\n👉 Please respond strictly in English.",
    "ja": base_prompt + "\n\n👉 必ず日本語で返答してください。韓国語や英語を含めないでください。必ず完全な日本語に翻訳してください。",
    "zh": base_prompt + "\n\n👉 请务必用中文作答。",
    "vi": base_prompt + "\n\n👉 Vui lòng chỉ trả lời bằng tiếng Việt."
}

# ====== 🔹 언어 감지 함수 ======
def detect_language(text):
    text = text.lower()
    if re.search(r'[\uac00-\ud7a3]', text): return 'ko'
    elif re.search(r'[\u3040-\u30ff]', text): return 'ja'
    elif re.search(r'[\u4e00-\u9fff]', text): return 'zh'
    elif re.search(r'[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹíìỉĩịóòỏõọúùủũụ]', text): return 'vi'
    else: return 'en'


# ====== 🔹 대화 이력 ======
conversation = []

# ====== 🔹 카테고리 키워드 및 데이터 로드 ======
# 공지사항 데이터 로드
df_notice = pd.read_csv(NOTICE_CSV_PATH, encoding="cp949")

# 카테고리 키워드 로드
CATEGORY_KEYWORDS_PATH = "/Users/shin/projects/CJ_AI/category_keywords.json"

with open(CATEGORY_KEYWORDS_PATH, "r", encoding="utf-8") as f:
    category_keywords = json.load(f)

# ====== 🔹 공지 제목 번역 함수 ======
async def translate_titles(titles, target_lang):
    if target_lang == 'ko':
        return titles

    lang_name_map = {
        'en': 'English',
        'ja': 'Japanese',
        'zh': 'Chinese',
        'vi': 'Vietnamese'
    }
    lang_name = lang_name_map.get(target_lang, 'English')

    # ====== 일본어 번역 시 사전 치환 적용 ======
    if target_lang == 'ja':
        replace_map_ja = {
            "마일리지": "マイレージ",
            "풀브라이트": "フルブライト",
            "기숙사": "寮",
            "창업": "起業",
            "장학금": "奨学金",
        }
        titles = [reduce(lambda t, kv: t.replace(*kv), replace_map_ja.items(), t) for t in titles]

    prompt = (
        f"You are a professional translator.\n"
        f"Translate the following Korean bulletin titles into fluent and complete {lang_name}.\n"
        f"🚫 Do NOT include any Korean, romanization, or untranslated words.\n"
        f"✅ Every line must be fully translated to {lang_name}, without exception.\n"
        f"❗ Even if the meaning is unclear, provide your best translation in {lang_name}.\n"
        f"❗ Especially for Japanese, absolutely no Korean should appear in the result.\n"
        f"Return only the translated titles, one per line, same order.\n"
        f"Titles:\n"
    )
    prompt += "\n".join([f"- {t}" for t in titles])

    async with aiohttp.ClientSession() as session:
        async with session.post(OLLAMA_URL, json={
            "model": "gemma3:4b",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }) as response:
            if response.status == 200:
                try:
                    content = await response.json()
                    content = content["message"]["content"]
                    lines = content.strip().split("\n")
                    cleaned = [line.strip().lstrip("*-• ") for line in lines if line.strip() and not line.lower().startswith("translation")]
                    while len(cleaned) < len(titles):
                        cleaned.append(titles[len(cleaned)])
                    return cleaned
                except Exception as e:
                    print("⚠️ 번역 파싱 오류:", e)
                    return titles
            return titles
    # 동기식 요청으로 변경
    # 2023-10-05: aiohttp를 사용한 비동기 요청에서 동기식으로 변경

# 임베딩 모델 로드 (다국어 지원)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def calculate_user_scores_semantic(user_input):
    input_emb = model.encode([user_input])
    scores = {}
    for category, keywords in category_keywords.items():
        # 카테고리 키워드 집합을 하나의 문장으로 합침
        keywords_text = ' '.join(keywords)
        kw_emb = model.encode([keywords_text])
        # 코사인 유사도 계산
        sim = cosine_similarity(input_emb, kw_emb)[0][0]
        scores[category] = sim
    # 상위 3개 카테고리 반환
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]

# ====== 🔹 공지 검색 ======
async def search_notice(category, lang='ko'):
    if category in df_notice.columns:
        top5 = df_notice[df_notice[category] > 0.0].sort_values(by=category, ascending=False).head(5)
        if top5.empty:
            return f"'{category}' 카테고리에는 현재 공지가 없습니다."

        titles = top5['공지 제목'].tolist()
        translated_titles = await translate_titles(titles, lang)

        response = f"[{category}] 카테고리의 점수 상위 5개 공지사항입니다:\n"
        for idx, (_, row) in enumerate(top5.iterrows()):
            title = translated_titles[idx] if idx < len(translated_titles) else row['공지 제목']
            response += f"- {title} ({row['링크']})\n"
        return response
    else:
        return f"'{category}' 카테고리를 찾을 수 없습니다."

# ====== 🔹 챗봇 응답 처리 ======
async def ask_ollama(question):
    lang = detect_language(question)
    category_scores = calculate_user_scores_semantic(question)
    if category_scores[0][1] > 0:
        top_category = category_scores[0][0]
        print(f"📚 관련 카테고리: {top_category}")
        result = await search_notice(top_category, lang)
        print(result)
        return result
    
    now_str = get_current_datetime_string(lang)

    system_prompt = (
        MULTILANG_PROMPTS.get(lang, MULTILANG_PROMPTS['en']) +
        f"\n\n현재 날짜와 시간은 반드시 참고하세요: {now_str}\n"
        "모든 답변은 반드시 이 날짜/시간을 기준으로 작성하세요."
    )

    messages = [{"role": "system", "content": system_prompt}] + conversation + [{"role": "user", "content": question}]

    async with aiohttp.ClientSession() as session:
        async with session.post(OLLAMA_URL, json={
            "model": "gemma3:4b",
            "messages": messages,
            "stream": True
        }) as response:
            if response.status == 200:
                print("🤖 챗봇:")
                full_response = ""
                async for line in response.content.iter_any():
                    if line:
                        data = line.decode('utf-8')
                        if '"content":"' in data:
                            content = data.split('"content":"')[1].split('"')[0].replace('\\n', '\n')
                            full_response += content
                            print(content, end="", flush=True)
                conversation.append({"role": "user", "content": question})
                conversation.append({"role": "assistant", "content": full_response})
                print()
                return full_response
            else:
                print("❌ 오류 발생:", response.text)
                return None
# 동기식 요청으로 변경
# 2023-10-05: aiohttp를 사용한 비동기 요청에서 동기식으로 변경
    
# ====== 🔹 로그인용 챗봇 응답 처리 ======
async def loginUser_ask_ollama(data):
    question = data.get("question")
    username = data.get("username")
    major = data.get("major")
    grade = data.get("grade")
    language = data.get("language")

    # 1. 국가 정보로 답변 언어 결정
    nationality_lang_map = {
        "한국어": "ko",
        "일본어": "ja",
        "중국어": "zh",
        "베트남어": "vi",
        "영어": "en",
    }
    lang = nationality_lang_map.get(language, "ko") # 기본값 한국어

    # 언어 이름 매핑
    lang_names = {
        "ko": "한국어",
        "en": "English",
        "ja": "日本語",
        "zh": "中文",
        "vi": "Tiếng Việt"
    }

    # 각 언어별 응답 지시문 
    lang_instruction = {
        "ko": "👉 모든 응답은 반드시 한국어로 작성해 주세요.",
        "en": "👉 Please respond strictly in English.",
        "ja": "👉 必ず日本語で返答してください。",
        "zh": "👉 请务必用中文作答。",
        "vi": "👉 Vui lòng chỉ trả lời bằng tiếng Việt."
    }.get(lang, "👉 Please respond strictly in English.")

    now_str = get_current_datetime_string(lang)

    # 명확한 프롬프트 구성
    custom_prompt = (
        f"\n\n현재 날짜와 시간은 반드시 참고하세요: {now_str}\n"
        "모든 답변은 반드시 이 날짜/시간을 기준으로 작성하세요."
        f"{base_prompt}\n\n"
        f"IMPORTANT: User language is {language}, so you MUST respond in {lang_names.get(lang, 'English')} regardless of the question language.\n\n"
        f"{lang_instruction}\n\n"
        f"사용자 정보:\n"
        f"- 이름: {username}\n"
        f"- 학년: {grade}\n"
        f"- 학과: {major}\n"
        f"- 국적: {language}\n"
        f"👉 반드시 {grade}학년 {major} 학생에게 적합한 정보를 제공하세요.\n"
    )

    # 3. 메시지 구성 및 Ollama 호출
    messages = [
        {"role": "system", "content": custom_prompt},
        {"role": "user", "content": question}
    ]

    async with aiohttp.ClientSession() as session:
        async with session.post(OLLAMA_URL, json={
            "model": "gemma3:4b",
            "messages": messages,
            "stream": False
        }) as response:
            if response.status == 200:
                content = await response.json()
                return content["message"]["content"]
            else:
                return "챗봇 서버 오류가 발생했습니다."
# 동기식 요청으로 변경
# 2023-10-05: aiohttp를 사용한 비동기 요청에서 동기식으로 변경