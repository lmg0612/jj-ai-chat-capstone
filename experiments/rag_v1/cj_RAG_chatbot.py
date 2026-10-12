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
import os

# 새로운 Chroma 패키지 사용
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# ====== 🔹 설정 ======
OLLAMA_URL = "http://localhost:11434/api/chat"
NOTICE_CSV_PATH = "/Users/shin/projects/CJ_AI_test/일반_카테고리_점수환산(6-4-0).csv"
SYSTEM_PROMPT_PATH = "/Users/shin/projects/CJ_AI_test/system_prompt.txt"
CHROMA_DB_PATH = "/Users/shin/projects/CJ_AI_test/chroma_db"

# ====== 🔹 누락된 함수들 추가 ======
def detect_language(text):
    """간단한 언어 감지"""
    if re.search(r'[가-힣]', text):
        return 'ko'
    elif re.search(r'[ひらがなカタカナ]', text):
        return 'ja'
    elif re.search(r'[一-龯]', text):
        return 'zh'
    elif re.search(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', text):
        return 'vi'
    else:
        return 'en'

async def translate_titles(titles, target_lang='ko'):
    """제목 번역 (간단한 버전)"""
    if target_lang == 'ko':
        return titles  # 한국어면 그대로 반환
    
    # 다른 언어면 번역 API 호출 (여기서는 간단히 원본 반환)
    return titles

# ====== 🔹 키워드 검색 함수 ======
def keyword_search(question, df):
    """질문에 포함된 키워드로 CSV에서 직접 공지 필터링"""
    # 키워드 분류 JSON 파일 로드
    with open('/Users/shin/projects/CJ_AI/category_keywords.json', encoding='utf-8') as f:
        category_keywords = json.load(f)
    # 모든 카테고리의 키워드 리스트 합치기
    all_keywords = sum(category_keywords.values(), [])
    # 질문에 포함된 키워드 추출
    matched_keywords = [kw for kw in all_keywords if kw in question]
    if not matched_keywords:
        # 질문에 직접 키워드가 없으면 모든 키워드로 공지 필터링
        matched_keywords = all_keywords
    # 공지 제목/본문에 키워드가 포함된 행만 반환
    mask = df['공지 제목'].astype(str).str.contains('|'.join(matched_keywords), case=False, na=False) | \
           df.get('본문 내용', pd.Series([""]*len(df))).astype(str).str.contains('|'.join(matched_keywords), case=False, na=False)
    return df[mask]


# ====== 🔹 ChromaDB 초기화 ======
def initialize_chroma_db():
    """CSV 데이터를 ChromaDB에 임베딩하여 저장"""
    
    try:
        # 기존 DB가 있으면 로드
        if os.path.exists(CHROMA_DB_PATH):
            print("📚 기존 ChromaDB 로드 중...")
            embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://localhost:11434")
            vector_db = Chroma(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=embeddings
            )
            return vector_db
        
        print("🔄 새로운 ChromaDB 생성 중...")
        
        # CSV 데이터 로드
        df_notice = pd.read_csv(NOTICE_CSV_PATH, encoding="cp949")
        
        # Ollama 임베딩 모델 설정
        embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://localhost:11434")
        
        # 문서 생성 (공지 제목 + 본문 내용)
        documents = []
        metadatas = []
        
        for _, row in df_notice.iterrows():
            # 공지 제목과 본문을 합쳐서 문서로 만듦
            content = f"제목: {row['공지 제목']}\n내용: {str(row['본문 내용'])}"
            documents.append(content)
            
            # 메타데이터 저장
            metadata = {
                "title": row['공지 제목'],
                "link": row['링크'],
                "department": row['부서'],
                "date": row['등록일']
            }
            metadatas.append(metadata)
        
        # ChromaDB에 저장
        vector_db = Chroma.from_texts(
            texts=documents,
            embedding=embeddings,
            metadatas=metadatas,
            persist_directory=CHROMA_DB_PATH
        )
        
        print(f"✅ {len(documents)}개 공지사항이 ChromaDB에 저장되었습니다.")
        return vector_db
        
    except Exception as e:
        print(f"❌ ChromaDB 초기화 오류: {e}")
        return None

# ChromaDB 초기화
vector_db = initialize_chroma_db()

# ====== 🔹 RAG 기반 답변 생성 ======
async def rag_generate_answer(question, lang='ko'):
    """검색된 공지를 context로 활용해 ollama가 답변 생성"""
    try:
        if vector_db is None:
            return "데이터베이스 초기화 오류가 발생했습니다."
        
        # 1. 관련 공지 검색 (임베딩 기반)
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(question)
        
        # ---- [수정] 임베딩 결과 없을 때 키워드 기반 검색 추가 ----
        if not relevant_docs:
            # CSV 데이터 로드
            df_notice = pd.read_csv(NOTICE_CSV_PATH, encoding="cp949")
            keyword_hits = keyword_search(question, df_notice)
            if not keyword_hits.empty:
                # 상위 3개만 추출
                keyword_hits = keyword_hits.head(3)
                context = ""
                for i, (_, row) in enumerate(keyword_hits.iterrows(), 1):
                    context += f"공지 {i}:\n"
                    context += f"제목: {row['공지 제목']}\n"
                    context += f"부서: {row['부서']}\n"
                    context += f"링크: {row['링크']}\n"
                    context += f"내용: {str(row.get('본문 내용', ''))[:200]}...\n\n"
                # 언어별 프롬프트
                lang_instruction = {
                    "ko": "한국어로만 답변하세요. 반드시 한국어 외 다른 언어를 섞지 마세요.",
                    "en": "Answer ONLY in English. Do NOT use any other language.",
                    "ja": "必ず日本語のみで答えてください。他の言語を絶対に混ぜないでください。",
                    "zh": "请只用中文回答。不要使用其他语言。",
                    "vi": "Chỉ trả lời bằng tiếng Việt. Tuyệt đối không sử dụng ngôn ngữ khác."
                }.get(lang, "한국어로만 답변하세요. 반드시 한국어 외 다른 언어를 섞지 마세요.")
                prompt = f"""다음 공지사항들을 참고해서 사용자의 질문에 정확하게 답변해주세요.
반드시 제공된 공지사항 정보(특히 링크)를 활용하여, 추가로 정보를 생성하지 마세요.

공지사항 정보:
{context}

사용자 질문: {question}

{lang_instruction}

※ 답변에는 반드시 관련 공지사항의 링크를 포함하세요.
예시: "2025학년도 장학금 안내는 아래 링크를 참고하세요: 
https://school.ac.kr/notice/12345"
"""
                messages = [{"role": "user", "content": prompt}]
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
                            return "답변 생성 중 오류가 발생했습니다."
            else:
                return "관련된 공지사항을 찾을 수 없습니다."
        # ---- [수정 끝] ----

        # 2. (임베딩 결과가 있을 때) Context 구성
        context = ""
        for i, doc in enumerate(relevant_docs, 1):
            metadata = doc.metadata
            context += f"공지 {i}:\n"
            context += f"제목: {metadata['title']}\n"
            context += f"부서: {metadata['department']}\n"
            context += f"링크: {metadata['link']}\n"
            context += f"내용: {doc.page_content[:200]}...\n\n"
        
        # 3. 언어별 프롬프트 구성
        lang_instruction = {
            "ko": "한국어로 답변해주세요.",
            "en": "Please respond in English.",
            "ja": "日本語で返答してください。",
            "zh": "请用中文作答。",
            "vi": "Vui lòng trả lời bằng tiếng Việt."
        }.get(lang, "한국어로 답변해주세요.")
        
        # 4. Ollama에 전송할 프롬프트
        prompt = f"""다음 공지사항들을 참고해서 사용자의 질문에 정확하게 답변해주세요.
                반드시 제공된 공지사항 정보만을 사용하고, 추가로 정보를 생성하지 마세요.
                
                공지사항 정보:
                {context}
                
                사용자 질문: {question}
                
                {lang_instruction}
                """
        messages = [{"role": "user", "content": prompt}]
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
                    return "답변 생성 중 오류가 발생했습니다."
                    
    except Exception as e:
        print(f"❌ RAG 답변 생성 오류: {e}")
        return f"오류가 발생했습니다: {str(e)}"

# ====== 🔹 메인 함수들 ======
async def ask_ollama(question):
    """기존 함수를 RAG 방식으로 대체"""
    lang = detect_language(question)
    result = await rag_generate_answer(question, lang)
    return result

async def loginUser_ask_ollama(data):
    """로그인 사용자용 RAG 기반 답변"""
    try:
        print(f"🔍 받은 데이터: {data}")  # 디버깅용 로그 추가
        
        question = data.get("question")
        username = data.get("username", "사용자")
        major = data.get("major", "")
        grade = data.get("grade", "")
        language = data.get("language", "한국어")

        print(f"🔍 질문: {question}")  # 디버깅용 로그 추가
        print(f"🔍 언어: {language}")  # 디버깅용 로그 추가

        # 언어 매핑
        nationality_lang_map = {
            "한국어": "ko",
            "일본어": "ja",
            "중국어": "zh",
            "베트남어": "vi",
            "영어": "en",
        }
        lang = nationality_lang_map.get(language, "ko")

        print(f"🔍 매핑된 언어: {lang}")  # 디버깅용 로그 추가

        # RAG 방식으로 답변 생성
        rag_result = await rag_generate_answer(question, lang)
        
        print(f"🔍 RAG 결과: {rag_result}")  # 디버깅용 로그 추가

        # 개인화된 인사말 추가
        personalized_result = f"안녕하세요 {username}님! ({grade}학년 {major})\n\n{rag_result}"
        
        return personalized_result
        
    except Exception as e:
        print(f"❌ 로그인 사용자 답변 오류: {e}")
        print(f"❌ 오류 상세: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()  # 전체 스택 트레이스 출력
        return f"오류가 발생했습니다: {str(e)}"
