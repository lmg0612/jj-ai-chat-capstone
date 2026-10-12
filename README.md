# JJ-AI_Chat — 전주대 스마트 캠퍼스 챗봇

> 2025 캡스톤디자인 · 팀장 · 학사 일정, 공지사항, 장학, 학식, 도서관 정보 등을 자연어로 안내하는 캠퍼스 챗봇

## 개요
학생들이 학사 일정·공지·시설 정보를 찾느라 여러 페이지를 돌아다니는 불편을 줄이기 위해, 크롤링한 학교 공지 데이터와 정리한 캠퍼스 정보를 바탕으로 질문에 답하는 챗봇을 만들었습니다.

## 동작 방식
1. **질문 분류**: 키워드 매칭 + SBERT 임베딩 유사도로 카테고리(학사/장학/학과/학식/도서관/동아리/연락처 등) 판단
2. **정보 조회**: 카테고리별 JSON 지식(`chatbot/prompt/*.json`)과 공지 CSV에서 후보 추출
3. **벡터 검색**: Ollama 임베딩 + Chroma DB로 관련 공지 검색
4. **답변 생성**: 로컬 LLM(Ollama)에 시스템 프롬프트와 근거를 넣어 답변 생성, 다국어 질문은 번역 후 처리

## 구조
```
chatbot/                  # 최종 버전
├─ result_test_real_final.py
├─ notice_data_utf8_test.csv   # 크롤링한 공지 데이터
└─ prompt/                     # 시스템 프롬프트 + 카테고리/학과/장학/학식/도서관 지식 JSON
experiments/
├─ chat_api/              # 초기 Flask API 서버, TF-IDF·NB 분류 실험
└─ rag_v1/                # KeyBERT·SBERT 기반 RAG 1차 버전
docs/                     # 개발보고서, 프롬프트 설계 메모
```

## 기술 스택
Python · Ollama · LangChain · Chroma · Sentence-Transformers · KeyBERT · pandas · Flask

## 실행
```bash
pip install pandas requests sentence-transformers scikit-learn langchain-chroma langchain-ollama aiohttp deep-translator
ollama serve   # 로컬 LLM·임베딩 모델 준비
python chatbot/result_test_real_final.py
```

## 내 역할
- 팀장: 기획·일정 관리, 개발보고서 작성
- 질문 분류 로직, 프롬프트 설계, 캠퍼스 지식 JSON 구축, RAG 파이프라인 구현

## 참고
팀 레포([JJNU-GMCI/CJ_AI](https://github.com/JJNU-GMCI/CJ_AI))와 별도로 만든 개인 포트폴리오 정리본입니다.
