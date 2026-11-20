# JJ-AI_Chat — 전주대 스마트 캠퍼스 챗봇

> 2025 캡스톤디자인 (2025.02 ~ 2025.08, 7인) · **팀장**
> 학사 일정, 공지사항, 장학, 학식, 도서관, 동아리, 연락처 정보를 자연어로 안내하는 캠퍼스 챗봇

학생들이 학사 일정·공지·시설 정보를 찾느라 학교 홈페이지 여러 곳을 돌아다니는 불편을 줄이기 위해, 크롤링한 공지 데이터와 정리한 캠퍼스 정보를 근거로 질문에 답하는 챗봇을 만들었습니다.

## 핵심 요약

- 질문을 먼저 **카테고리로 분류**(키워드 매칭 + KR-SBERT 유사도)해 찾을 범위를 좁힙니다.
- 학식·도서관·학사 일정·연락처·동아리처럼 **정형화된 정보는 JSON 지식에서 바로 응답**하고, 공지 관련 질문은 **필터링 + RAG**로 답합니다.
- RAG는 Ollama 임베딩(`mxbai-embed-large`) + Chroma 벡터 DB로 공지 3건을 찾고, 로컬 LLM(`gemma3:4b`)이 **공지 정보만 근거로** 답변합니다.
- 한국어·영어·일본어·중국어·베트남어 질문은 **한국어로 번역해 같은 흐름으로 처리**하고, 답변을 질문 언어로 되돌립니다.

## 아키텍처

```
사용자 질문 (ko / en / ja / zh / vi)
  │
  ├─ 언어 감지 (유니코드 문자 범위 규칙) ─▶ 한국어로 번역 (deep-translator)
  │
  ├─ 인사·정체성 질문 ─────────▶ 고정 응답
  ├─ 도서관 / 지도 / 학식 / 연락처 / 동아리 ─▶ 캠퍼스 지식 JSON 조회
  ├─ "N월 학사일정" ───────────▶ academic_schedule_parsed.json
  │
  └─ 공지·장학 질문
        ├─ 카테고리 분류: 키워드 매칭 → KR-SBERT 유사도 (snunlp/KR-SBERT-V40K-klueNLI-augSTS)
        ├─ 공지 CSV 필터링: 카테고리 키워드 + SBERT 의미 필터 (상위 후보)
        └─ RAG: Chroma(mxbai-embed-large, k=3) ─▶ 시스템 프롬프트 + 근거 공지 ─▶ Ollama gemma3:4b
  │
  └─ 답변을 질문 언어로 번역해 반환
```

## 구현 포인트

| 영역 | 구현 내용 | 관련 파일 |
|---|---|---|
| 질문 분류 | 카테고리 키워드 사전 매칭 후, 실패 시 KR-SBERT 임베딩 유사도로 카테고리 판단 | `chatbot/result_test_real_final.py` (`keyword_match_category`, `get_top_category_sbert`) |
| 정형 정보 응답 | 학식 메뉴, 도서관 운영시간·휴관일, 캠퍼스 지도, 학사 일정, 부서 연락처, 중앙동아리를 JSON에서 조회 | `chatbot/prompt/*.json` |
| 공지 검색 | 카테고리 키워드 필터 → SBERT 의미 필터 → 모집 현황(모집중·선착순·정보 없음)별 정리 | `semantic_filter`, `keyword_filter` |
| RAG 답변 | Chroma에서 공지 3건 검색, 제목·부서·링크·본문 일부를 근거로 넣고 "공지 정보만 사용" 규칙으로 생성 | `rag_generate_answer` |
| 다국어 | 문자 범위 규칙으로 5개 언어 감지, Google 번역(deep-translator)으로 입력·출력 변환, 언어별 기본 메시지 사전 | `detect_language`, `translate_to_korean`, `DEFAULT_MESSAGES` |
| 프롬프트 | 답변 범위와 근거 사용 규칙을 정한 시스템 프롬프트 | `chatbot/prompt/system_prompt.txt`, `docs/prompt-design/` |

## 개발 과정과 개선

테스트하며 드러난 문제를 줄이기 위해 모델과 구조를 여러 번 바꿨습니다.

| 단계 | 구성 | 코드 |
|---|---|---|
| 1차 | Flask API 서버 + TF-IDF·나이브 베이즈 질문 분류, Ollama 연동 테스트 | `experiments/chat_api/` |
| 2차 | KeyBERT·SBERT 키워드 검색 + Chroma RAG | `experiments/rag_v1/` |
| 최종 | 키워드 + KR-SBERT 분류, 카테고리별 지식 JSON, Chroma RAG, 다국어 번역 처리 | `chatbot/` |

주로 부딪힌 문제는 **공지 분류 오류, 질문 의도 오해, 다국어 질문 오역**이었습니다. 모델 교체, 파인튜닝, RAG 도입을 차례로 시도하며 개선했고, 팀 내부 테스트에서 기존 학교 챗봇보다 나은 답변을 확인했습니다. 별도의 정량 평가 기록은 남기지 않았습니다.

## 프로젝트 구조

```
chatbot/                         최종 버전
├─ result_test_real_final.py     질문 분류 · 지식 조회 · 공지 필터 · RAG · 다국어 처리
├─ notice_data_utf8_test.csv     크롤링한 공지 데이터
└─ prompt/                       시스템 프롬프트, 카테고리·학과·장학 키워드, 학사일정·학식·도서관·동아리 JSON
experiments/
├─ chat_api/                     1차: Flask API, TF-IDF·NB 분류 실험
└─ rag_v1/                       2차: KeyBERT·SBERT + Chroma RAG
docs/                            개발보고서, 프롬프트 설계 메모
```

## 빠른 실행

```bash
pip install pandas requests sentence-transformers scikit-learn langchain-chroma langchain-ollama aiohttp deep-translator
ollama pull gemma3:4b && ollama pull mxbai-embed-large
ollama serve
python chatbot/result_test_real_final.py
```

## 기술 스택

Python · Ollama (gemma3:4b, mxbai-embed-large) · LangChain · Chroma · Sentence-Transformers (KR-SBERT) · KeyBERT · scikit-learn · pandas · deep-translator · Flask

## 내 역할

- **팀장**: 기획·일정 관리, 전체 파트 확인·조율, 개발보고서 작성
- 질문 분류 로직, 프롬프트 설계, 캠퍼스 지식 JSON 구축, RAG 파이프라인 구현

## 공개 범위

팀 레포([JJNU-GMCI/CJ_AI](https://github.com/JJNU-GMCI/CJ_AI))와 별도로 만든 개인 포트폴리오 정리본입니다. Chroma DB 파일은 포함하지 않았습니다.
