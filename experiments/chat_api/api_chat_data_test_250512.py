import requests
import pandas as pd

# 시스템 프롬프트 불러오기 (최초 1회만 로드)
with open(r"C:\Users\dlakr\OneDrive\바탕 화면\chat_api\prompt\system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

# 초기 대화 설정
conversation = [{"role": "system", "content": system_prompt}]
conversation.append({"role": "system", "content": "모든 응답은 지정된 프롬프트 규칙에 따라야 합니다."})

OLLAMA_URL = "http://localhost:11434/api/chat"
NOTICE_CSV_PATH = r"C:\Users\dlakr\OneDrive\바탕 화면\chat_api\일반_카테고리_점수환산.csv"
df_notice = pd.read_csv(NOTICE_CSV_PATH)

# 🌐 지원 언어 목록
LANGUAGES = {
    "ko": "한국어",
    "en": "English",
    "ja": "日本語",
    "zh": "中文",
    "vi": "Tiếng Việt"
}

# 🌐 언어 선택 초기화
def select_language():
    print("\n🌐 Please select your language:")
    for code, name in LANGUAGES.items():
        print(f"  [{code}] {name}")
    
    while True:
        user_choice = input("\n💬 Enter language code: ").strip().lower()
        if user_choice in LANGUAGES:
            # 🗣️ 시작 메시지
            start_message = {
                "ko": "안녕하세요! Chat JJ입니다! 'exit' 입력 시 종료.\n",
                "en": "Hello! This is Chat JJ! Type 'exit' to quit.\n",
                "ja": "こんにちは！Chat JJです！'exit'と入力して終了します。\n",
                "zh": "你好！这是 Chat JJ！输入 'exit' 以退出。\n",
                "vi": "Xin chào! Đây là Chat JJ! Nhập 'exit' để thoát.\n"
            }
            print(f"\n🗣️ 언어가 '{LANGUAGES[user_choice]}'로 설정되었습니다.\n")
            print(start_message[user_choice])
            return user_choice
        else:
            print("❌ 지원하지 않는 언어입니다. 다시 선택해주세요.")

# 사용자 언어 선택
current_language = select_language()

# 🌐 Gemma 모델 번역 함수 (새로 추가)
def gemma_translate(text, target_lang="ko"):
    try:
        # 번역 프롬프트 생성
        translation_prompt = f"Translate this to {LANGUAGES[target_lang]}: {text}"
        response = requests.post(OLLAMA_URL, json={
            "model": "gemma3:4b",
            "messages": [{"role": "user", "content": translation_prompt}],
            "stream": False
        })

        if response.status_code == 200:
            translated_text = response.json().get("content", text)
            # 첫 글자 대문자로 수정
            return translated_text.capitalize()
        else:
            print("❌ 번역 오류:", response.text)
            return text
    except Exception as e:
        print("❌ 번역 예외 발생:", e)
        return text

# 🔹 고정형 메시지 번역 처리
def get_fixed_response(prompt_key):
    fixed_responses = {
        "self_intro": "안녕하세요! 저는 전주대학교 학생들을 위한 캠퍼스 도우미 챗봇 'JJ'입니다. 궁금한 정보가 있으시면 언제든지 말씀해주세요!",
        "tab_prompt": "어떤 탭의 공지사항을 확인하시겠어요? [일반, 학사, 장학, 등록, 구매/입찰, 채용] 중에서 선택해주세요.",
        "error_response": "죄송합니다. 해당 기능은 지원하지 않습니다.",
        "unknown_response": "죄송합니다. 잘 모르겠습니다. 다시 한번 말씀해 주세요."
    }
    
    # 현재 언어로 번역하여 반환
    if current_language == "ko":
        return fixed_responses[prompt_key]
    else:
        return gemma_translate(fixed_responses[prompt_key], current_language)

# 🔹 공지 데이터 번역 처리
def search_notice(category):
    if category in df_notice.columns:
        top5 = df_notice[df_notice[category] > 0.3].sort_values(by=category, ascending=False).head(5)
        if top5.empty:
            return get_fixed_response("error_response")
        
        response = f"[{category}] 카테고리의 점수 상위 5개 공지사항입니다:\n"
        for _, row in top5.iterrows():
            title = row['공지 제목']
            link = row['링크']
            # 번역 처리
            translated_title = gemma_translate(title, current_language)
            response += f"- {translated_title} ({link})\n"
        return response
    else:
        return get_fixed_response("error_response")

# 🔹 메인 챗봇 로직
def ask_ollama(question):
    conversation.append({"role": "user", "content": question})
    response = requests.post(OLLAMA_URL, json={
        "model": "gemma3:4b",
        "messages": conversation,
        "stream": True
    }, stream=True)

    if response.status_code == 200:
        print("🤖 챗봇:")
        full_response = ""
        for line in response.iter_lines():
            if line:
                data = line.decode('utf-8')
                if '"content":"' in data:
                    content_part = data.split('"content":"')[1].split('"')[0]
                    content_part = content_part.replace('\\n', '\n')
                    full_response += content_part

        # 🔄 번역 처리
        final_response = gemma_translate(full_response, current_language)
        print(final_response)
        conversation.append({"role": "assistant", "content": final_response})
        print()
        return final_response
    else:
        print("❌ 오류 발생:", response.text)

# 🏁 시작 메시지 (언어 선택 포함)
while True:
    user_input = input("👤 나: ")
    if user_input.lower() in ['종료','exit', 'quit']:
        print("👋 챗봇을 종료합니다.")
        break
    ask_ollama(user_input)
