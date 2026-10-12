import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# CSV 파일 로드
df = pd.read_csv("categorized_notices_advanced.csv")

# 텍스트 구성 및 라벨 선택
df["text"] = df["공지 제목"].fillna('') + " " + df["본문 내용"].fillna('')
df_labeled = df[["text", "카테고리"]].dropna()

# 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(df_labeled["text"], df_labeled["카테고리"], test_size=0.2, random_state=42)

# TF-IDF + 나이브베이즈 파이프라인 구성
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', MultinomialNB()),
])

# 모델 학습
pipeline.fit(X_train, y_train)

# 모델 저장
joblib.dump(pipeline, "updated_model.joblib")
