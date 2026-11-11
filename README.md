# Mini-RAG «Аккорд»

Небольшой RAG-прототип для ответов на вопросы по правилам творческого центра. Поиск сочетает BM25 и dense-эмбеддинги `multilingual-e5-small` в FAISS, а результаты объединяются через RRF.

## Режимы работы

- `chunk` — возвращает найденные фрагменты документов без генерации.
- `llm` — передаёт найденные фрагменты в контекст `Qwen/Qwen3-0.6B` и возвращает краткий ответ.

В обоих режимах отдельно возвращаются ID документа и дословная цитата. Архивные документы не попадают в индекс.

## Запуск

Рекомендуется Python 3.10–3.12.

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --reload
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Откройте <http://127.0.0.1:8000>. Swagger доступен на <http://127.0.0.1:8000/docs>.

При первом запуске будет скачана dense-модель. Qwen загружается при первом запросе в режиме `llm`; на CPU генерация может занять некоторое время.

## Evals

```bash
python evaluate.py --mode chunk
python evaluate.py --mode llm
```

Результаты сохраняются в `results_chunk.csv` и `results_llm.csv`. Оба режима проходят 7/7 проверочных вопросов.
