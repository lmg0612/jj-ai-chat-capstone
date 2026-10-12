import requests
import pandas as pd
import re
import json
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import aiohttp
import asyncio
from deep_translator import GoogleTranslator
from datetime import datetime, timedelta

# === 경로 설정 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_PATH = os.path.join(BASE_DIR, "prompt", "system_prompt.txt")
CSV_PATH = os.path.join(BASE_DIR, "notice_data_utf8_test.csv")
CATEGORY_PATH = os.path.join(BASE_DIR, "prompt", "category_keywords.json")
SCHOLARSHIP_PATH = os.path.join(BASE_DIR, "prompt", "scholarship_keywords_filtered.json")
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")    
DEPARTMENT_PATH = os.path.join(BASE_DIR, "prompt", "department_keywords.json")
SCHEDULE_PATH = os.path.join(BASE_DIR, "prompt", "academic_schedule_parsed.json")
JJ_MENU_PATH = os.path.join(BASE_DIR, "prompt", "jeonju_university_menu.json")
JJ_LIBRARY_PATH = os.path.join(BASE_DIR, "prompt", "jj_library.json")
CLUB_PATH = os.path.join(BASE_DIR, "prompt", "central_clubs_data.json")
OLLAMA_URL = "http://localhost:11434/api/chat"

# === 데이터 로딩 ===
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()
with open(CATEGORY_PATH, "r", encoding="utf-8") as f:
    category_keywords = json.load(f)
with open(SCHOLARSHIP_PATH, "r", encoding="utf-8") as f:
    scholarship_keywords = json.load(f)
with open(SCHEDULE_PATH, "r", encoding="utf-8") as f:
    academic_schedule = json.load(f)
with open(DEPARTMENT_PATH, "r", encoding="utf-8") as f:
    department_keywords = json.load(f)
# with open(DEPARTMENT_PATH, "r", encoding="utf-8") as f:
#     dept_dict = json.load(f)
with open(JJ_MENU_PATH, "r", encoding="utf-8") as f:
    jj_menu = json.loads( f.read())
with open(JJ_LIBRARY_PATH, "r", encoding="utf-8") as f:
    jj_library = json.load(f)
with open(CLUB_PATH, "r", encoding="utf-8") as f:
    club_data = json.load(f)
df_notice = pd.read_csv(CSV_PATH, encoding="utf-8")

# === 모델 초기화 ===
encoder = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# === 다국어 메시지 사전 ===
DEFAULT_MESSAGES = {
    "no_notice": {
        "ko": "❗ 관련된 공지사항을 찾지 못했습니다.",
        "en": "❗ No relevant notices found.",
        "ja": "❗ 関連するお知らせが見つかりませんでした。",
        "zh": "❗ 未找到相关公告。",
        "vi": "❗ Không tìm thấy thông báo liên quan."
    },
    "rag_error": {
        "ko": "❌ RAG 답변 생성 중 오류가 발생했습니다.",
        "en": "❌ An error occurred while generating a RAG response.",
        "ja": "❌ RAG応答の生成中にエラーが発生しました。",
        "zh": "❌ 生成 RAG 响应时出错。",
        "vi": "❌ Đã xảy ra lỗi khi tạo phản hồi RAG."
    },
    "greeting": {
        "ko": "안녕하세요! 저는 전주대학교 학생들을 위한 캠퍼스 도우미 챗봇 'JJ'입니다.\n",
        "en": "Hello! I’m ‘JJ’, your campus assistant chatbot for Jeonju University students.\n",
        "ja": "こんにちは！私は全州大学の学生のためのキャンパスアシスタントチャットボット「JJ」です。\n",
        "zh": "你好！我是为全州大学学生提供服务的校园助手聊天机器人“JJ”。\n",
        "vi": "Xin chào! Tôi là chatbot trợ lý khuôn viên ‘JJ’ dành cho sinh viên Đại học Jeonju.\n"
    }
}

# === 언어 감지 ===
def detect_language(text):
    if re.search(r'[\uac00-\ud7a3]', text): return 'ko'
    elif re.search(r'[\u3040-\u30ff]', text): return 'ja'
    elif re.search(r'[\u4e00-\u9fff]', text): return 'zh-CN' #zh-TW 추가
    elif re.search(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', text.lower()): return 'vi'
    else: return 'en'

# === 입력 번역 (비한국어 → 한국어)
def translate_to_korean(text, lang):
    if lang == "ko":
        return text
    
    try:
        # 언어 코드 매핑 (GoogleTranslator 지원 형식으로 변환)
        lang_mapping = {
            "zh": "zh-CN",  # 중국어 간체로 매핑
            "zh-cn": "zh-CN",
            "zh-tw": "zh-TW",
            "en": "en",
            "ja": "ja",
            "vi": "vi"
        }
        
        # 매핑된 언어 코드 사용
        mapped_lang = lang_mapping.get(lang.lower(), lang)
        
        translator = GoogleTranslator(source=mapped_lang, target="ko")
        result = translator.translate(text)
        return result
        
    except Exception as e:
        print(f"❌ 번역 실패({lang}) → {mapped_lang} --> {e}")
        return text  # 번역 실패 시 원문 그대로 사용
    
def translate_from_korean(text, target_lang):
    """한국어 텍스트를 목표 언어로 번역"""
    try:
        if target_lang == "ko":
            return text
        
        # 언어 코드 매핑
        lang_mapping = {
            "zh": "zh-CN",  # 중국어 간체로 매핑
            "zh-cn": "zh-CN",
            "zh-tw": "zh-TW",
            "en": "en",
            "ja": "ja",
            "vi": "vi"
        }
        
        mapped_lang = lang_mapping.get(target_lang.lower(), target_lang)
        
        # GoogleTranslator를 사용한 번역
        translator = GoogleTranslator(source='ko', target=mapped_lang)
        result = translator.translate(text)
        return result
        
    except Exception as e:
        print(f"번역 중 오류 발생: {e}")
        return text  # 번역 실패시 원문 반환

# === 인사 여부 감지 ===
def is_greeting_or_identity(q):
    patterns = ["안녕", "누구", "역할", "정체", "뭐해", "하는일", "챗봇", "너는"]
    return any(p in q.lower() for p in patterns)

def is_general_notice_request(q):
    base_keywords = ["공지", "공지사항", "커뮤니티"]
    return any(b in q.lower() for b in base_keywords)

def get_schedule_by_month(query: str):
    query = query.replace(" ", "").replace("\u00a0", "")
    match = re.search(r"(\d{1,2})월", query)
    if not match:
        return None
    month = int(match.group(1))
    matched = []
    for item in academic_schedule:
        start_month = int(item["시작일"].split("-")[1])
        end_month = int(item["종료일"].split("-")[1])
        if month == start_month or month == end_month:
            matched.append(f"- {item['제목']} (행사기간: {item['행사기간']})")
    if matched:
        return f"[{month}월 학사일정 안내]\n" + "\n".join(matched)
    return f"📭 {month}월에는 등록된 학사일정이 없습니다."

# === 세부 조건 질문 여부 판단 ===
#case 1
# def is_detailed_query(q):
#     base = ["전공", "학과", "AI", "소프트웨어", "프로그래밍", "컴퓨터", "공학"]
#     qualifier = ["관련", "대상", "조건", "참여", "지원", "누가", "어떤"]

#     q = q.lower()

#     base_match = any(re.search(b, q) for b in base)
#     qualifier_match = any(re.search(qf, q) for qf in qualifier)

#     return base_match and qualifier_match

#test_case 2
def is_detailed_query(q):
    q = q.lower()

    department_terms = [kw.lower() for sublist in department_keywords.values() for kw in sublist]

    base_match = any(dept in q for dept in department_terms)
    qualifier_keywords = ["관련", "대상", "조건", "참여", "지원", "누가", "어떤"]
    qualifier_match = any(qf in q for qf in qualifier_keywords)

    return base_match and qualifier_match

# === 의미 기반 카테고리 분류 ===
def get_top_category_sbert(question, encoder, category_keywords):
    q_vec = encoder.encode([question])[0]
    best_cat = None
    best_score = 0
    for cat, keywords in category_keywords.items():
        combined = " ".join(keywords)
        cat_vec = encoder.encode([combined])[0]
        score = cosine_similarity([q_vec], [cat_vec])[0][0]
        if score > best_score:
            best_cat = cat
            best_score = score
    return best_cat if best_score > 0.4 else None

def keyword_match_category(query, category_keywords):
    for cat, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword in query:
                return cat
    return None

# === 연락처 정보 ===
def get_contact_info(question):
    lower_q = question.lower().replace(" ", "")
    
    # 연락처/전화번호 관련 키워드 정의
    contact_keywords = ["연락처", "전화번호", "전화", "연락", "문의", "상담", "담당자", "연락망", "콜센터", "고객센터"]
    
    for keyword in contact_keywords:
        if keyword in lower_q:
            print(f"🔹 Step 3: 연락처 키워드 '{keyword}' 감지 → 연락처 안내 페이지 링크 제공")
            # 언어별 메시지 정의 (URL은 번역하지 않음)
            url = "https://www.jj.ac.kr/jj/colleges/colleges-Introduction.do"
            
            if lang == "ko":
                return f"[연락처 안내 페이지]: {url}\n"
            elif lang == "en":
                return f"[Contact Information Page]: {url}\n"
            elif lang == "ja":
                return f"[連絡先情報ページ]: {url}\n"
            elif lang in ["zh", "zh-CN", "zh-TW"]:
                return f"[联系信息页面]: {url}\n"
            elif lang == "vi":
                return f"[Trang thông tin liên hệ]: {url}\n"
            else:
                return f"[Contact Information Page]: {url}\n"
    
    return None

CLUBS_BY_GROUP = club_data["clubs_by_group"]

def get_club_list_by_query(query: str) -> str:
    query_clean = query.lower().replace(" ", "")

    # 전체 중앙동아리 요청
    if "동아리" in query or ("중앙" in query and "전체" in query):
        result = "[전주대 중앙동아리 목록]\n"
        for group, clubs in CLUBS_BY_GROUP.items():
            result += f"🔸 {group} 분과:\n  - " + ", ".join(clubs) + "\n"
        return result
    return None

# === 전주대학교 메뉴 정보 ===
def get_jj_menu_info(question):
    """사용자 질문에 따라 전주대학교 메뉴 정보를 반환하는 함수"""
    
    lower_q = question.lower().replace(" ", "")
    
    # 메뉴 키워드 정의
    menu_keywords = ["메뉴", "식단", "급식", "식사", "밥", "점심", "저녁", "학식", "급식표", "식사시간", "식당", 
                     "구내식당", "학생식당", "학생회관", "학생식당메뉴", "학생회관메뉴", "중식", "석식", "점심밥", "저녁밥"]
    
    # 요일 키워드 정의
    weekday_keywords = {
        "월": ["월", "월요일", "monday"],
        "화": ["화", "화요일", "tuesday"],
        "수": ["수", "수요일", "wednesday"],
        "목": ["목", "목요일", "thursday"],
        "금": ["금", "금요일", "friday"]
    }
    
    # 식사 시간 키워드 정의
    meal_keywords = {
        "중식": ["중식", "점심", "lunch", "점심밥"],
        "석식": ["석식", "저녁", "dinner", "저녁밥"]
    }
    
    # 메뉴 관련 키워드가 있는지 확인
    if not any(keyword in lower_q for keyword in menu_keywords):
        return None
    
    try:
        # 특정 요일 질문인지 확인
        target_day = None
        for day, keywords in weekday_keywords.items():
            if any(keyword in lower_q for keyword in keywords):
                target_day = day
                break
        
        # 특정 식사 시간 질문인지 확인
        target_meal = None
        for meal, keywords in meal_keywords.items():
            if any(keyword in lower_q for keyword in keywords):
                target_meal = meal
                break
        
        # 응답 생성
        if target_day and target_meal:
            # 특정 요일의 특정 식사
            if target_day in jj_menu and target_meal in jj_menu[target_day]:
                meal_info = jj_menu[target_day][target_meal]
                response = f"## {target_day}요일 {target_meal} 메뉴\n"
                response += f"**시간**: {meal_info['time']}\n\n"
                response += "**메뉴**:\n"
                for i, menu_item in enumerate(meal_info['menu'], 1):
                    response += f"{i}. {menu_item}\n"
                return response
            else:
                return f"{target_day}요일 {target_meal} 메뉴 정보를 찾을 수 없습니다."
        
        elif target_day:
            # 특정 요일의 전체 메뉴
            if target_day in jj_menu:
                response = f"## {target_day}요일 전체 메뉴\n\n"
                
                # 중식
                if '중식' in jj_menu[target_day]:
                    lunch_info = jj_menu[target_day]['중식']
                    response += f"### 🍽️ 중식 ({lunch_info['time']})\n"
                    for i, menu_item in enumerate(lunch_info['menu'], 1):
                        response += f"{i}. {menu_item}\n"
                    response += "\n"
                
                # 석식
                if '석식' in jj_menu[target_day]:
                    dinner_info = jj_menu[target_day]['석식']
                    response += f"### 🍽️ 석식 ({dinner_info['time']})\n"
                    for i, menu_item in enumerate(dinner_info['menu'], 1):
                        response += f"{i}. {menu_item}\n"
                
                return response
            else:
                return f"{target_day}요일 메뉴 정보를 찾을 수 없습니다."
        
        elif target_meal:
            # 전체 요일의 특정 식사
            response = f"## 이번 주 {target_meal} 메뉴\n\n"
            weekdays = ['월', '화', '수', '목', '금']
            
            for day in weekdays:
                if day in jj_menu and target_meal in jj_menu[day]:
                    meal_info = jj_menu[day][target_meal]
                    response += f"### {day}요일 ({meal_info['time']})\n"
                    for menu_item in meal_info['menu']:
                        response += f"• {menu_item}\n"
                    response += "\n"
            
            return response
        
        else:
            # 전체 메뉴
            response = "## 🍽️ 전주대학교 이번 주 전체 메뉴\n\n"
            weekdays = ['월', '화', '수', '목', '금']
            
            for day in weekdays:
                if day in jj_menu:
                    response += f"### {day}요일\n"
                    
                    # 중식
                    if '중식' in jj_menu[day]:
                        lunch_info = jj_menu[day]['중식']
                        response += f"**중식** ({lunch_info['time']}): "
                        response += ", ".join(lunch_info['menu']) + "\n\n"
                    
                    # 석식
                    if '석식' in jj_menu[day]:
                        dinner_info = jj_menu[day]['석식']
                        response += f"**석식** ({dinner_info['time']}): "
                        response += ", ".join(dinner_info['menu']) + "\n\n"
            
            return response
            
    except Exception as e:
        return f"메뉴 정보를 불러오는 중 오류가 발생했습니다: {str(e)}"
    
def extract_keywords_from_question(question, dept_dict):
    matched_keywords = []
    for dept_name, keywords in dept_dict.items():
        for kw in keywords:
            if kw in question:
                matched_keywords.append(kw)
    return list(set(matched_keywords))  # 중복 제거

# # === 유사도 기반 공지 필터링 ===
# def semantic_filter(question, df, topn=5, keyword_list=None):
#     q_vec = encoder.encode([question])
#     scores = []

#     for _, row in df.iterrows():
#         t = str(row.get("공지 제목", ""))
#         b = str(row.get("본문 내용", ""))
#         o = str(row.get("이미지 텍스트", ""))

#         t_sim = cosine_similarity(encoder.encode([t]), q_vec)[0][0]
#         b_sim = cosine_similarity(encoder.encode([b]), q_vec)[0][0] if b.strip() else 0
#         o_sim = cosine_similarity(encoder.encode([o]), q_vec)[0][0] if o.strip() else 0

#         score = 0.2 * t_sim + 0.8 * b_sim + 0.4 * o_sim

#         # 🔍 키워드 기반 점수 보정
#         if keyword_list:
#             keyword_bonus = 0
#             for kw in keyword_list:
#                 if kw in t or kw in b or kw in o:
#                     keyword_bonus += 0.4  # 한 키워드당 0.2점 가산
#             score += keyword_bonus

#         scores.append(score)

#     top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topn]
#     return df.iloc[top_idx].drop_duplicates(subset=["공지 제목"]).reset_index(drop=True)

def semantic_filter(translated_q, original_q, df, topn=5):
    # ✅ 학과 키워드 기준으로 먼저 필터링 (원문 + 번역문 모두 기준)
    department_terms = [kw.lower() for sublist in department_keywords.values() for kw in sublist]
    all_text = (translated_q + " " + original_q).lower()
    matched_depts = [dept for dept in department_terms if dept in all_text]

    if matched_depts:
        print(f"[학과 필터링 적용] 질문에 포함된 학과 키워드: {matched_depts}")
        df = df[df.apply(
            lambda row: any(
                dept in (str(row["공지 제목"]) + str(row["본문 내용"])).lower()
                for dept in matched_depts
            ),
            axis=1
        )]

    # ✅ 의미 유사도 기반 필터링
    q_vec = encoder.encode([translated_q])
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
    return df.iloc[top_idx].drop_duplicates(subset=["공지 제목"]).reset_index(drop=True)

# === 키워드 기반 단순 필터링 ===
def keyword_filter(df, category, lang="ko"):
    now = pd.Timestamp.now()
    one_month_ago = now - pd.Timedelta(days=30)

    if category not in df.columns:
        return f"📌 [{category}] 카테고리 정보를 찾을 수 없습니다.\n"

    filtered = df[df[category] > 0].copy()

    if "등록일" not in filtered.columns:
        return f"⚠️ '등록일' 컬럼이 없어 날짜 필터를 건너뜁니다."

    filtered["등록일"] = pd.to_datetime(filtered["등록일"], errors="coerce")
    filtered = filtered[filtered["등록일"] >= one_month_ago]
    filtered = filtered.drop_duplicates(subset=["공지 제목", "링크"])

    모집중 = filtered[filtered["모집 현황"] == "모집중"]
    선착순 = filtered[filtered["모집 현황"] == "선착순"]
    정보없음 = filtered[filtered["모집 현황"] == "정보 없음"]

    lang_mapping = {
        "zh-cn": "zh-CN", "zh-tw": "zh-TW",
        "en": "en", "ja": "ja", "vi": "vi"
    }
    mapped_lang = lang_mapping.get(lang.lower(), lang)

    result = f"[{category}]에 대한 공지입니다. (한 달 이내)\n\n"
    total = 0

    def append_notices(df_part, label):
        nonlocal result, total
        if total >= 5:
            return
        count = 0
        for _, row in df_part.iterrows():
            if total >= 5:
                break
            title = row["공지 제목"]
            try:
                if lang != "ko":
                    title = GoogleTranslator(source='ko', target=mapped_lang).translate(title)
            except Exception as e:
                print(f"❌ 번역 실패: {e}")
            if count == 0:
                result += f"📌 [{label}] 공지\n"
            result += f"[{title}]: {row['링크']}\n"
            total += 1
            count += 1
        if count > 0:
            result += "\n"

    append_notices(모집중, "모집중")
    append_notices(선착순, "선착순")
    append_notices(정보없음, "정보 없음")

    if total == 0:
        return {
            "ko": "📭 최근 한 달 이내의 관련 공지가 없습니다.",
            "en": "📭 No relevant notices in the past month.",
            "ja": "📭 過去1か月以内に関連するお知らせはありません。",
            "zh": "📭 最近一个月内没有相关公告。",
            "vi": "📭 Không có thông báo liên quan trong tháng qua."
        }.get(lang, "📭 최근 한 달 이내의 관련 공지가 없습니다.")

    return result.strip()

def get_jj_map(question):
    """전주대학교 캠퍼스 맵 링크 제공"""
    lower_q = question.lower().replace(" ", "")
    
    # 위치/지도 관련 키워드 정의
    contact_keywords = ["지도", "캠퍼스맵", "캠퍼스지도", "약도", "건물위치", "위치", "길찾기", "캠퍼스", "맵", "위치정보", "건물찾기", "전주대지도", "전주대학교지도", "전주대캠퍼스맵", 
                        "전주대학교캠퍼스맵", "전주대약도", "전주대학교약도", "전주대건물위치", "전주대학교건물위치", "전주대길찾기", "전주대학교길찾기", "전주대캠퍼스", "전주대학교캠퍼스", "전주대맵", 
                        "전주대학교맵", "전주대위치정보", "전주대학교위치정보", "전주대건물찾기", "전주대학교건물찾기", "전주대캠퍼스지도", "전주대학교캠퍼스지도", "전주대캠퍼스맵링크", "전주대학교캠퍼스맵링크", 
                        "전주대약도링크", "전주대학교약도링크", "전주대건물위치링크", "전주대학교건물위치링크", "전주대길찾기링크", "전주대학교길찾기링크", "몇층", "건물", "위치찾기", "위치안내", "건물안내", 
                        "건물정보", "건물찾기", "전주대위치", "전주대학교위치", "어디에", "어디", "위치", "어디에있는건물", "어디에있는건물위치", "어디에있는건물찾기", "어디에있는건물안내", "어디에있는건물정보", 
                        "공대건물", "예술관", "학생회관", "학생회관위치", "학생회관안내", "학생회관정보", "학생회관찾기", "학생회관위치정보", "학생회관건물위치", "학생회관건물안내", "학생회관건물정보",
                        "학생회관건물찾기", "학생회관건물위치정보", "학생회관건물안내정보", "학생회관건물안내링크", "학생회관건물정보링크", "학생회관건물찾기링크", "어디로", "어디로가야", "어디로가는건물", "어디로가는건물위치",
                        "어디로가는건물찾기", "어디로가는건물안내", "어디로가는건물정보", "어디로가는건물위치정보", "어디로가는건물안내링크", "어디로가는건물정보링크", "건물", "빌딩",
                        "몇호", "몇호실"]
    
    for keyword in contact_keywords:
        if keyword in lower_q:
            print(f"🔹 Step 3: 위치/지도 키워드 '{keyword}' 감지 → 전주대 지도 페이지 링크 제공")
            # 언어별 메시지 정의 (URL은 번역하지 않음)
            url = "https://www.jj.ac.kr/jj/introduction/campus-map.do?referringCampus=1&campus=1&type=BUILDING"
            
            if lang == "ko":
                return f"[전주대 지도]: {url}\n"
            elif lang == "en":
                return f"[Jeonju University Campus Map]: {url}\n"
            elif lang == "ja":
                return f"[全州大学キャンパスマップ]: {url}\n"
            elif lang in ["zh", "zh-CN", "zh-TW"]:
                return f"[全州大学校园地图]: {url}\n"
            elif lang == "vi":
                return f"[Bản đồ khuôn viên Đại học Jeonju]: {url}\n"
            else:
                return f"[Jeonju University Campus Map]: {url}\n"
    
    return None

def get_jj_library_info(question):
    """전주대학교 도서관 정보 제공"""
    lower_q = question.lower().replace(" ", "")
    
    # 도서관 관련 키워드 정의
    library_keywords = ["도서관", "루디아홀", "스터디룸", "열람실", "도서", "책", "자료실", "열람", "대출", "반납", "도서검색", "전자책", "인쇄책", "학술정보관", "학술정보", "학술자료",
                        "대여", "도서관정보", "도서관안내", "도서관검색", "도서관대출", "도서관반납", "도서관열람실", "도서관스터디룸", "도서관자료실", 
                        "책검색", "책대출", "책반납", "책열람실", "책스터디룸", "책자료실", "책정보", "책안내", "전자책검색", "전자책대출", "전자책반납",
                        "인쇄책검색", "인쇄책대출", "인쇄책반납", "미디어밸리", "hatch", "해치", "고서실", "특수자료실", "노트북존"]
    
    # 운영시간 관련 키워드 정의
    operating_keywords = ["운영시간", "개관시간", "열람시간", "대출시간", "반납시간", "도서관운영시간", "도서관개관시간", "도서관열람시간", "도서관대출시간", "도서관반납시간",
                         "몇시", "몇시까지", "몇시에", "몇시에열림", "몇시에닫힘", "몇시에닫나요", "몇시에열리나요", "몇시에닫히나요",
                         "언제까지", "언제까지열림", "언제까지닫힘", "언제까지닫나요", "언제까지열리나요", "언제까지닫히나요", 
                         "몇시부터", "몇시부터열림", "몇시부터닫힘", "몇시부터닫나요", "몇시부터열리나요", "몇시부터닫히나요",
                         "언제부터", "언제부터열림", "언제부터닫힘", "언제부터닫나요", "언제부터열리나요", "언제부터닫히나요", 
                         "운영", "시간", "언제", "개방", "이용시간", "사용시간"]
    
    # 기간 관련 키워드 정의
    period_keywords = {
        "학기중": ["학기중", "학기", "수업기간", "정규학기", "개강", "학기중운영"],
        "방학중": ["방학중", "방학", "방학기간", "휴학기간", "방학중운영"],
        "시험기간": ["시험기간", "시험", "기말시험", "중간시험", "시험기간중", "시험주간"]
    }
    
    # 특정 시설 키워드 정의
    facility_keywords = {
        "일반자료실_및_3층자료실": ["일반자료실", "3층자료실", "자료실", "일반"],
        "스터디룸": ["스터디룸", "스터디", "그룹스터디", "팀스터디"],
        "대출_반납": ["대출", "반납", "대여", "책대출", "책반납"],
        "미디어밸리": ["미디어밸리", "미디어", "밸리"],
        "HATCH_Lounge": ["hatch", "해치", "라운지", "lounge"],
        "고서실_특수자료실": ["고서실", "특수자료실", "고서", "특수자료"],
        "자유열람실_노트북존": ["자유열람실", "노트북존", "자유열람", "노트북", "24시간"]
    }

    # 도서관 키워드 확인
    has_library_keyword = any(keyword in lower_q for keyword in library_keywords)
    has_operating_keyword = any(keyword in lower_q for keyword in operating_keywords)

    # 도서관 키워드가 없으면 None 반환
    if not has_library_keyword:
        return None
    
    # 도서관 정보와 운영시간 가져오기
    library_info = jj_library.get("전주대학교_도서관", {})
    operating_hours = library_info.get("운영시간", {})

    if has_library_keyword and has_operating_keyword:
            print(f"🔹 Step 3: 도서관 운영시간 키워드 감지 → 도서관 운영시간 제공")
            
            # 기간 확인
            detected_period = None
            for period, keywords in period_keywords.items():
                if any(keyword in lower_q for keyword in keywords):
                    detected_period = period
                    break
            
            # 특정 시설 확인
            detected_facility = None
            for facility, keywords in facility_keywords.items():
                if any(keyword in lower_q for keyword in keywords):
                    detected_facility = facility
                    break
            
            return format_library_hours(operating_hours, library_info.get("휴관일", []), 
                                      detected_period, detected_facility)
    elif has_library_keyword:
        print(f"🔹 Step 3: 도서관 키워드 감지 → 도서관 정보 페이지 링크 제공")
        return get_basic_library_info(lang)
    
def format_library_hours(operating_hours, holidays, period=None, facility=None):
    """도서관 운영시간을 포맷팅하여 반환"""
    
    # 기본 제목 (한국어)
    response = "## 📚 전주대학교 도서관 운영시간\n\n"
    
    # 특정 기간이 지정된 경우
    if period and period in operating_hours:
        period_data = operating_hours[period]
        response += f"### {period}\n\n"
        
        # 특정 시설이 지정된 경우
        if facility and facility.replace("_", " ") in period_data:
            facility_key = facility.replace("_", " ")
            response += f"**{facility}**\n"
            facility_data = period_data[facility_key]
            for time_key, time_value in facility_data.items():
                if time_key != "비고":
                    response += f"- {time_key}: {time_value}\n"
                else:
                    response += f"- 비고: {time_value}\n"
        else:
            # 모든 시설 표시
            for facility_key, facility_data in period_data.items():
                response += f"**{facility_key}**\n"
                for time_key, time_value in facility_data.items():
                    if time_key != "비고":
                        response += f"- {time_key}: {time_value}\n"
                    else:
                        response += f"- 비고: {time_value}\n"
                response += "\n"
    else:
        # 모든 기간 표시
        for period_key, period_data in operating_hours.items():
            response += f"### {period_key}\n\n"
            
            for facility_key, facility_data in period_data.items():
                response += f"**{facility_key}**\n"
                for time_key, time_value in facility_data.items():
                    if time_key != "비고":
                        response += f"- {time_key}: {time_value}\n"
                    else:
                        response += f"- 비고: {time_value}\n"
                response += "\n"
    
    # 휴관일 정보 추가
    if holidays:
        response += f"### 휴관일\n"
        for holiday in holidays:
            response += f"- {holiday}\n"
    
    return response
    
def get_basic_library_info(lang):
    """기본 도서관 정보 링크 제공"""
    url = "https://lib.jj.ac.kr/#/"
    
    if lang == "ko":
        return f"[전주대 도서관 정보]: {url}\n"
    elif lang == "en":
        return f"[Jeonju University Library Information]: {url}\n"
    elif lang == "ja":
        return f"[全州大学図書館情報]: {url}\n"
    elif lang in ["zh", "zh-CN", "zh-TW"]:
        return f"[全州大学图书馆信息]: {url}\n"
    elif lang == "vi":
        return f"[Thông tin Thư viện Đại học Jeonju]: {url}\n"
    else:
        return f"[Jeonju University Library Information]: {url}\n"

# === LangChain ChromaDB 초기화 ===
def initialize_chroma_db():
    if os.path.exists(CHROMA_DB_PATH):
        embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://localhost:11434")
        return Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    return None

vector_db = initialize_chroma_db()

# === LangChain 기반 RAG 응답 생성 ===
async def rag_generate_answer(question, lang='ko'):
    if vector_db is None:
        return "❌ Chroma DB가 초기화되지 않았습니다."

    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(question)

    if not relevant_docs:
        return DEFAULT_MESSAGES["no_notice"].get(lang, DEFAULT_MESSAGES["no_notice"]["ko"])


    context = ""
    for i, doc in enumerate(relevant_docs, 1):
        metadata = doc.metadata
        context += f"공지 {i}:\n"
        context += f"제목: {metadata['title']}\n"
        context += f"부서: {metadata['department']}\n"
        context += f"링크: {metadata['link']}\n"
        context += f"내용: {doc.page_content[:200]}...\n\n"

    lang_instruction = {
        "ko": "한국어로 답변해주세요.",
        "en": "Please respond in English.",
        "ja": "日本語で返答してください。",
        "zh": "请用中文作答。",
        "vi": "Vui lòng trả lời bằng tiếng Việt."
    }.get(lang, "한국어로 답변해주세요.")

    prompt = f"""다음 공지사항들을 참고해서 사용자의 질문에 정확하게 답변해주세요.
공지사항 정보만을 사용하고, 추가 정보를 생성하지 마세요.

공지사항 정보:
{context}

사용자 질문: {question}

{lang_instruction}"""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    async with aiohttp.ClientSession() as session:
        async with session.post(OLLAMA_URL, json={"model": "gemma3:4b", "messages": messages, "stream": False}) as response:
            if response.status == 200:
                content = await response.json()
                response_text = content["message"]["content"]
                if lang != "ko":
                    try:
                        response_text = GoogleTranslator(source="ko", target=lang).translate(response_text)
                    except Exception as e:
                        print(f"❌ RAG 결과 번역 실패: {e}")
                return response_text
            return DEFAULT_MESSAGES["rag_error"].get(lang, DEFAULT_MESSAGES["rag_error"]["ko"])
        

# === 메인 루프 ===
print("💬 전주대 챗봇 JJ (LangChain + SBERT + 카테고리 + LLM fallback 통합형)\n")

while True:
    q = input("👤 질문: ")
    if q.lower() in ["exit", "종료", "quit"]:
        print("👋 종료합니다.")
        break

    print(f"🔹 Step 1: 질문 수신됨 → {q}")
    lang = detect_language(q)
    print(f"🔹 Step 2: 감지된 언어: {lang}")

    translated_q = translate_to_korean(q, lang) if lang != "ko" else q
    if lang != "ko":
        print(f"🔹 Step 2.5: 번역된 질문 → {translated_q}")
    sbert_input = f"{translated_q} (원문: {q})" if lang != "ko" else translated_q

    if is_greeting_or_identity(translated_q):
        print("🔹 Step 3: 인사 또는 정체성 질문 감지 → 인사 응답")
        print(DEFAULT_MESSAGES["greeting"].get(lang, DEFAULT_MESSAGES["greeting"]["ko"]))
        continue

    # if is_department_contact_query(q):
    #     print("🔹 Step 3: 학과/부서 연락처 관련 질문 감지 → 링크 제공")
    #     print("[학과/부서 안내 페이지]: https://www.jj.ac.kr/jj/colleges/colleges-Introduction.do\n")
    #     continue

    # 전주대학교 도서관 정보 요청 감지
    jj_library_info = get_jj_library_info(translated_q)
    if jj_library_info:
        print("🔹 Step 3: 전주대학교 도서관 정보 요청 감지 → 도서관 정보 제공")
        # 언어별 출력 처리
        if lang != "ko":
            print(f"🔹 Step 3.5: 도서관 정보를 {lang}로 번역 중...")
            translated_library_info = translate_from_korean(jj_library_info, lang)
            print(translated_library_info)
        else:
            print(jj_library_info)
        continue

    # 전주대학교 캠퍼스 맵 정보 요청 감지
    jj_map_info = get_jj_map(translated_q)
    if jj_map_info:
        print("🔹 Step 3: 전주대학교 캠퍼스 맵 정보 요청 감지 → 캠퍼스 맵 링크 제공")
        print(jj_map_info)
        continue
    
    translated_q = translated_q.replace(" ", "")

    if "전체" in translated_q and ("학사일정" in translated_q or "학업 일정" in translated_q or "학사 일정" in translated_q or "학업일정" in translated_q or "일정" in translated_q):
        print("🔹 Step 3: 전체 학사일정 질문 감지 → 사이트 링크 제공")

        schedule_link = "https://www.jj.ac.kr/jj/campuslife/schedule.do?mode=list&&articleLimit=10&article.offset=0"

        schedule_msg_multilang = {
            "ko": f"[학사일정 전체 보기]: {schedule_link}\n",
            "en": f"[View Full Academic Calendar]: {schedule_link}\n",
            "ja": f"[全体の学事日程を見る]: {schedule_link}\n",
            "zh": f"[查看完整的学术日程]: {schedule_link}\n",
            "vi": f"[Xem toàn bộ lịch học]: {schedule_link}\n"
        }

        print(schedule_msg_multilang.get(lang, schedule_msg_multilang["ko"]))
        continue

    if ("학사일정" in translated_q or "학업 일정" in translated_q or "학사 일정" in translated_q or "학업일정" in translated_q or "일정" in translated_q) and re.search(r"\d{1,2}\s*월", translated_q):
        print("🔹 Step 3: 월별 학사일정 질문 감지 → 일정 필터링")
        schedule_result = get_schedule_by_month(translated_q)  # 기본은 한국어
        if lang != "ko":
            print(f"🔹 Step 3.5: 일정 정보를 {lang}로 번역 중...")
            schedule_result = translate_from_korean(schedule_result, lang)
        print(schedule_result)
        continue    

    matched_category = keyword_match_category(translated_q, category_keywords)

    # if matched_category:
    #     category = matched_category
    #     print(f"🔹 Step 3: 키워드 직접 매칭 결과 → {category}")
    # else:
    #     category = get_top_category_sbert(sbert_input, encoder, category_keywords)
    #     print(f"🔹 Step 3: SBERT 기반 유사도 분류 결과 → {category}")
    #     print(f"🔹 Step 3: 상위 카테고리: {category}")

    df_result = df_notice.copy()
    df_result = df_result.drop_duplicates(subset=["링크", "공지 제목"])

    lower_q = q.lower().replace(" ", "")
    category = None
    matched = False

    # 🔹 Step 1: 특수 장학 키워드 → 링크 제공
    for group, keywords in scholarship_keywords.items():
        for kw in keywords:
            if kw.replace(" ", "") in lower_q:
                print(f"🔹 Step 3: 특수 장학 키워드 '{kw}' 감지 → 링크 제공")
                print("[장학금 안내 페이지]: https://www.jj.ac.kr/jj/campuslife/scholarship01.do\n")
                matched = True
                break
        if matched:
            break

    if matched:
        continue  # 여기서 종료

    # 🔹 Step 2: 장학공지 키워드 감지 → 카테고리 강제 설정
    for cat, keywords in category_keywords.items():
        if cat.startswith("장학공지"):
            for kw in keywords:
                if kw.replace(" ", "") in lower_q:
                    print(f"🔹 Step 3: 장학공지 키워드 '{kw}' 감지 → {cat} 카테고리로 설정")
                    category = cat
                    matched = True
                    break
        if matched:
            break

    # 🔹 Step 3: 일반 키워드 매칭
    if not matched:
        matched_category = keyword_match_category(translated_q, category_keywords)
        if matched_category:
            category = matched_category
            print(f"🔹 Step 3: 키워드 직접 매칭 결과 → {category}")
            matched = True

    # 연락처 정보 요청 감지
    contact_info = get_contact_info(translated_q)
    if contact_info:
        print("🔹 Step 3: 연락처 키워드 감지 → 연락처 안내내 페이지 링크 제공")
        print(contact_info)
        continue

    # 전주대학교 메뉴 정보 요청 감지
    jj_menu_info = get_jj_menu_info(translated_q)
    if jj_menu_info:
        print("🔹 Step 3: 전주대학교 메뉴 정보 요청 감지 → 메뉴 정보 제공")
        # 언어별 출력 처리
        if lang != "ko":
            print(f"🔹 Step 3.5: 메뉴 정보를 {lang}로 번역 중...")
            translated_menu_info = translate_from_korean(jj_menu_info, lang)
            print(translated_menu_info)
        else:
            print(jj_menu_info)
        continue

    # === ✅ 세부 조건 질문 먼저 처리 ===
    # case 1
    # if is_detailed_query(translated_q):
    #     print("🔸 세부 조건 질문 감지됨 → 의미기반 유사도 필터링")

    #     # 기본적으로 전체 데이터 사용
    #     filtered_df = df_notice

    #     if category:
    #         print(f"🔸 [{category}] 기준 상위 50개 대상 → 유사도 필터링")
    #         filtered_df = df_notice[df_notice[category] > 0].sort_values(by=category, ascending=False).head(50)

    #     # 의미기반 필터링 (여기서 refined가 결과임)
    #     refined = semantic_filter(sbert_input, filtered_df, topn=5)

    #     # 중복 제거 (링크 + 공지 제목 기준)
    #     refined = refined.drop_duplicates(subset=["공지 제목"])

    #     if refined.empty:
    #         print("❗ 관련 공지 없음 → RAG로 응답 생성 중...")
    #         result = asyncio.run(rag_generate_answer(q, lang))
    #         print(f"🤖 {result}")
    #     else:
    #         print("🔎 유사도 기반 필터링 결과:")
    #         for _, row in refined.iterrows():
    #             title = row["공지 제목"]
    #             link = row["링크"]
    #             if lang != "ko":
    #                 try:
    #                     title = GoogleTranslator(source='ko', target=lang).translate(title)
    #                 except Exception as e:
    #                     print(f"❌ 제목 번역 실패: {e}")
    #             print(f"[{title}]: {link}")
    #     continue

    if is_detailed_query(translated_q):
        print("🔸 세부 조건 질문 감지됨 → 의미기반 유사도 필터링")

        # 전체 공지에서 시작
        filtered_df = df_notice

        if category:
            print(f"🔸 [{category}] 기준 상위 50개 대상 → 유사도 필터링")
            filtered_df = df_notice[df_notice[category] > 0.4].sort_values(by=category, ascending=False).head(50)
            print("🔎 상위 50개 공지 목록:")
            for idx, row in filtered_df.iterrows():
                print(f"- {row['공지 제목']} ({row['등록일']}) → {row['링크']}")

        # 의미기반 필터링
        # keyword_list = extract_keywords_from_question(translated_q, dept_dict)
        # refined = semantic_filter(translated_q, df_notice, topn=20, keyword_list=keyword_list)  # 5개보다 넉넉히 뽑아야 모집중/선착순 분류 가능
        refined = semantic_filter(translated_q, q, filtered_df, topn=20)
        refined = refined.drop_duplicates(subset=["공지 제목", "링크"])

        if refined.empty:
            print("❗ 관련 공지 없음 → RAG로 응답 생성 중...")
            result = asyncio.run(rag_generate_answer(q, lang))
            print(f"🤖 {result}")
        else:
            print("🔎 유사도 기반 공지 필터링 완료 → 상태 기반 출력")
            # ▶ 이 부분이 keyword_filter_flexible 로직 사용부
            def filter_and_format_by_status(df_input):
                now = pd.Timestamp.now()
                one_month_ago = now - pd.Timedelta(days=60)
                df_input["등록일"] = pd.to_datetime(df_input["등록일"], errors="coerce")
                df_input = df_input[df_input["등록일"] >= one_month_ago]
                df_input = df_input.drop_duplicates(subset=["공지 제목", "링크"])

                모집중 = df_input[df_input["모집 현황"] == "모집중"]
                선착순 = df_input[df_input["모집 현황"] == "선착순"]
                정보없음 = df_input[df_input["모집 현황"] == "정보 없음"]

                lang_mapping = {
                    "zh-cn": "zh-CN", "zh-tw": "zh-TW",
                    "en": "en", "ja": "ja", "vi": "vi"
                }
                mapped_lang = lang_mapping.get(lang.lower(), lang)

                result = ""
                total = 0

                def append(df_part, label):
                    nonlocal result, total
                    if total >= 5:
                        return
                    count = 0
                    for _, row in df_part.iterrows():
                        if total >= 5:
                            break
                        title = row["공지 제목"]
                        link = row["링크"]
                        try:
                            if lang != "ko":
                                title = GoogleTranslator(source='ko', target=mapped_lang).translate(title)
                        except Exception as e:
                            print(f"❌ 번역 실패: {e}")
                        if count == 0:
                            result += f"📌 [{label}] 공지\n"
                        result += f"[{title}]: {link}\n"
                        total += 1
                        count += 1
                    if count > 0:
                        result += "\n"

                append(모집중, "모집중")
                append(선착순, "선착순")
                append(정보없음, "정보 없음")

                return result.strip() if total > 0 else {
                    "ko": "📭 최근 한 달 이내의 관련 공지가 없습니다.",
                    "en": "📭 No relevant notices in the past month.",
                    "ja": "📭 過去1か月以内に関連するお知らせはありません。",
                    "zh": "📭 最近一个月内没有相关公告。",
                    "vi": "📭 Không có thông báo liên quan trong tháng qua."
                }.get(lang, "📭 최근 한 달 이내의 관련 공지가 없습니다.")

            print(filter_and_format_by_status(refined))

        continue

    club_result = get_club_list_by_query(q)
    if club_result:
        print("🔹 Step 3: 중앙동아리 질문 감지 → 리스트 출력")
        print(club_result)
        continue

    # === ✅ 카테고리 없음인 경우 ==
    if category is None:
        if is_general_notice_request(translated_q):
            print("📢 일반 공지 요청 감지 → 전체 공지 상위 5개 출력")
            top5 = df_notice.head(5)
            for _, row in top5.iterrows():
                print(f"[{row['공지 제목']}]: {row['링크']}")
        else:
            print("❗ 관련 카테고리를 찾을 수 없습니다. → RAG로 질문 전달")
            result = asyncio.run(rag_generate_answer(q, lang))
            print(f"🤖 {result}")
        continue

    # === 카테고리 있음 + 세부질문 아님 ===
    print("🔸 단순 공지 요청 감지 → 키워드 기반 Top5 출력")
    # if lang == "ko":
    #     print(keyword_filter(df_notice, category))
    # else:
    #     print(keyword_filter_translated(df_notice, category, lang))

    # filtered_result = keyword_filter(df_notice, category, lang="ko")
    filtered_result = keyword_filter(df_notice, category, lang="ko")  # 원문(한국어) 결과 생성

    if lang != "ko":
        filtered_result = translate_from_korean(filtered_result, lang)  # 번역 적용
    print(filtered_result)
