# ==========================================================================
# chat_rubert.py — sentiment-классификатор на базе ruBERT (fine-tuned)
# ==========================================================================
# Запуск вручную: uv run python chat_rubert.py
# Импорт из web.py: from chat_rubert import load_model, predict
# ==========================================================================

from pathlib import Path
from typing import Any, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Путь к весам — всегда рядом с этим файлом, независимо от cwd.
MODEL_DIR = Path(__file__).parent / "rubert-classifier-full"

MAX_LENGTH = 64

POSITIVE_LABEL_ID = 1


def load_model(path: Path = MODEL_DIR) -> Tuple[Any, Any, torch.device]:
    """Загружает ruBERT и возвращает (model, tokenizer, device)."""
    if not path.is_dir():
        raise FileNotFoundError(
            f"Не найдена папка модели: {path}\n"
            f"Убедись, что rubert-classifier-full лежит рядом с chat_rubert.py."
        )

    device = torch.device("cpu")

    print(f"Загружаю ruBERT из {path}...")
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForSequenceClassification.from_pretrained(str(path))
    model.to(device)
    model.eval()
    print(f"Модель загружена. Device: {device}")
    return model, tokenizer, device


def predict(text: str, model, tokenizer, device) -> float:
    """Возвращает вероятность ПОЗИТИВА (0.0..1.0)."""
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    if not text.strip():
        raise ValueError("text is empty")

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        prob_positive = probs[0, POSITIVE_LABEL_ID].item()
    return prob_positive


def main() -> None:
    """Консольный режим."""
    model, tokenizer, device = load_model()
    print("Введите фразу. Пустая строка / exit / quit / q — выход.\n")

    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text or text.lower() in {"exit", "quit", "q"}:
            break

        try:
            prob = predict(text, model, tokenizer, device)
        except ValueError as e:
            print(f"Ошибка: {e}\n")
            continue

        label = 1 if prob >= 0.5 else 0
        verdict = "ПОЗИТИВ" if label == 1 else "НЕГАТИВ"
        print(f"{verdict}  p(позитив)={prob:.3f}\n")


if __name__ == "__main__":
    main()