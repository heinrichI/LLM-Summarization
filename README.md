# LLM-Summarization
Суммаризация с помощью LLM.

## Файлы проекта

### Основные скрипты суммаризации

| Файл | Описание |
|------|----------|
| `SumByOpenAI-API_custom_map_reduce.py` | Кастомная map-reduce суммаризация через OpenAI-совместимый API. Три стадии: map (с переводом на русский, ≤100 слов) → reduce (промежуточное резюме) → final (итоговое резюме). Включает ручной контроль токенов, кастомное логирование. |
| `SumByOpenAI-API_langchain_map_reduce.py` | Вариант с использованием встроенных цепочек LangChain: `MapReduceDocumentsChain`, `ReduceDocumentsChain`. Меньше кода, автоматический коллапсинг через `token_max`. |
| `SumByLlamaCpp.py` | Суммаризация через LlamaCpp (локальная модель). |
| `SumTranslateByOpenAI-API.py` | Суммаризация с последующим переводом. |
| `TranslateByOpenAI-API.py` | Перевод текста через OpenAI-совместимый API. |

### Вспомогательные скрипты

| Файл | Описание |
|------|----------|
| `FileIO.py` | Чтение файлов и сохранение результатов. |
| `ConvertFb2.py` / `ConvertFb2_2.py` | Конвертация FB2 в текст. |
| `ConvertEpub.py` / `ConvertEpub2.py` / `ConvertEpub_ebooklib.py` | Конвертация EPUB в текст. |