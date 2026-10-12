from flask import Flask, request, jsonify
from tset import ask_ollama

app = Flask(__name__)

@app.route('/', methods=['POST'])
def home():
    return "Flask 서버가 정상적으로 실행 중입니다"

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json() # Spring Boot에서 전송한 JSON 데이터 가져오기
        question = data.get("question") # JSON 데이터에서 사용자 질문 추출

        if not question:
            return jsonify({"error": "질문이 전달되지 않았습니다"}), 400
        
        # Ollama에 질문전송
        response = ask_ollama(question)

        return jsonify({"response": response}) # 결과 반환
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", port = 5000) # Flask 서버 실행