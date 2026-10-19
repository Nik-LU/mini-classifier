# ==========================================================================
# web.py — Flask-сервер: принимает текст, возвращает prediction тональности
# ==========================================================================
# Запуск: uv run python web.py
# Открыть: http://localhost:5001
# Health: http://localhost:5001/health
# ==========================================================================

import os
import logging
from flask import Flask, request, jsonify, render_template

from chat_rubert import load_model, predict

# ---------- Логирование ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

app = Flask(__name__)

# ---------- Загрузка модели ОДИН раз при старте ----------
print("Загружаю модель...")
model, tokenizer, device = load_model()
print("Готово. Сервер стартует...")


# ---------- Логирование запросов к /predict ----------
@app.after_request
def log_predict_request(response):
    if request.path == "/predict" and request.method == "POST":
        try:
            text = (request.get_json(silent=True) or {}).get("text", "")
        except Exception:
            text = "?"
        app.logger.info(f"/predict <- {text!r}  -> {response.status_code}")
    return response


# ---------- Роуты ----------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    """Проверка, что сервер жив. Android может дёргать при старте."""
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "empty text"}), 400

    prob = predict(text, model, tokenizer, device)
    label = 1 if prob >= 0.5 else 0
    sentiment = "positive" if label == 1 else "negative"

    return jsonify({
        "text": text,
        "label": label,
        "sentiment": sentiment,
        "prob": prob,
    })


if __name__ == "__main__":
    # HOST и PORT можно переопределить через env.
    # Для теста с реального телефона в Wi-Fi: HOST=0.0.0.0 uv run python web.py
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5001"))
    app.run(host=host, port=port, debug=True)