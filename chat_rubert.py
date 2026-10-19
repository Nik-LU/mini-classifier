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
app.logger.info("Загружаю модель...")
model, tokenizer, device = load_model()
app.logger.info("Готово. Сервер стартует...")


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

    app.logger.info(f"/predict <- {text!r}")

    if not text:
        app.logger.info("/predict -> 400 empty text")
        return jsonify({"error": "empty text"}), 400

    try:
        prob = predict(text, model, tokenizer, device)
    except Exception as e:
        app.logger.error(f"Prediction failed: {e}")
        return jsonify({"error": "internal server error"}), 500

    label = 1 if prob >= 0.5 else 0
    sentiment = "positive" if label == 1 else "negative"

    app.logger.info(f"/predict -> 200 {sentiment} (p={prob:.3f})")

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
    port = int(os.environ.get("PORT", "5001")) # Оставили ваш порт!
    
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    
    app.run(host=host, port=port, debug=debug)