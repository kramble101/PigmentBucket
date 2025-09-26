# PigmentBucket — m0 Runbook

## Установка
```
cd ~/Documents/GitHub/PigmentBucket
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quick checks
```
python3 -m pip install -r requirements.txt
python3 -m pytest
./scripts/run_service.sh
python3 scripts/pigmentbucket_run.py --dry-run --limit 1 --http-timeout 60
python3 scripts/pigmentbucket_run.py --limit 3 --force-min-k 2 --min-duration-sec 2 --similarity-threshold 0.95 --locations-store sqlite --report-dir logs
python3 scripts/pigmentbucket_run.py --limit 3 --export-locations-only --locations-store json
python3 scripts/pigmentbucket_run.py --limit 3 --add-resolve-markers --locations-store sqlite --report-format both --min-duration-sec 3 --similarity-threshold 0.90
python3 scripts/pigmentbucket_gui.py
```

## Запуск сервиса
```
./scripts/run_service.sh
curl http://127.0.0.1:8765/health
```

## Прогон анализатора
```
export PIGMENT_SERVICE_URL="http://127.0.0.1:8765"
python3 scripts/pigmentbucket_run.py --dry-run --report-dir logs --http-timeout 60
python3 scripts/pigmentbucket_run.py --report-dir logs --force-min-k 2
python3 scripts/pigmentbucket_run.py --report-dir logs --ignore-cache
```

- Таймаут HTTP по умолчанию 60 с; можно переопределить через `--http-timeout` или `PIGMENT_HTTP_TIMEOUT`.
- `--force-min-k 2` включает эвристику «best-effort» раскраски, когда сервис нашёл только один кластер.
- `--ignore-cache` отключает файловый кэш фич (по умолчанию `.cache/features/<hash>.npz`).
- `--locations-store sqlite|json` выбирает хранилище локаций: SQLite (стабильные ID, файл `state/locations.sqlite`) или JSON (`state/locations/<job_id>.json`).
- `--export-locations-only` выполняет анализ и экспорт отчётов без применения цветов и маркеров.
- `--add-resolve-markers` добавляет маркеры начала/конца локаций на таймлайн Resolve (.undo хранит координаты маркеров).
- `--report-format json|csv|both` управляет набором артефактов, которые сервер сохраняет в `logs/` (по умолчанию both).
- `--undo-markers <run_id>` удаляет только маркеры (без восстановления цветов) на основе сохранённого undo-
  слепка.

## Selection
```
python3 scripts/pigmentbucket_run.py --selection all
python3 scripts/pigmentbucket_run.py --selection inout
python3 scripts/pigmentbucket_run.py --selection selected
```

Если в режиме `inout` не выставлены маркеры Mark In/Out, раннер предупредит и использует весь таймлайн. В режиме `selected` при отсутствии API для выбранных клипов произойдёт WARN и попытка использовать Mark In/Out; при отсутствии диапазона — возврат к анализу всего таймлайна. Логи CLI/GUI теперь всегда отображают `effective_mode`, общее количество клипов и имена первых элементов выборки, что упрощает проверку scoping.

## Где логи
Сервис сохраняет отчёты в logs/<job_id>.json (опционально CSV).

## Reports & Colors
- JSON отчёт соответствует схеме v1.0 и содержит стабильную палитру `palette_name: default_v1`.
- CSV (`logs/<job_id>.csv`) дублирует клиповую таблицу и подходит для табличного анализа.
- Пример CSV-строк:
```
job_id,clip_uid,clip_name,start_tc,end_tc,cluster_id,clip_color
abcd1234,clip-001,Intro,00:00:00:00,00:00:05:12,0,Cyan
abcd1234,clip-002,B-Roll,00:00:05:12,00:00:10:00,1,Yellow
```
- При повторных запусках одинаковые cluster_id получают те же clip_color.
- В summary доступны `feature_dim`, `chosen_k`, `silhouette`, `sampled_frames` для диагностики пайплайна.
- В summary дополнительно отображается `coloring` (например, `{"coloring": "skipped", "reason": "k==1"}`) и количество попаданий кэша `cache_hits`.
- CSV дополнен столбцами `location_id`, `persisted_location_id`; summary содержит `locations`, `locations_stats`, `location_store`, а при SQLite‑режиме также `locations_persisted`, `matched_locations`, `new_locations`, `centroid_dim`.

## Location grouping & persistence
- Последовательные клипы с одинаковым `cluster_id` и близкими фичами группируются в локации (`location_id`).
- Порог по длительности (`--min-duration-sec`, по умолчанию 3 с) и hysteresis-поглощение коротких вставок управляются через конфиг; похожие по косинусной метрике сегменты объединяются, регулируя `--similarity-threshold`.
- При использовании `--locations-store sqlite` локации получают стабильные `persisted_location_id`; повторные запуски сопоставляют центроиды и переиспользуют ID. Статистика `locations_persisted`, `matched_locations`, `new_locations` попадает в summary и логи GUI/CLI.
- `--locations-store json` сохраняет одноразовый отчёт без стабильных ID (полезно для анализа без записи в SQLite).
- JSON/CSV отчёты дополняются блоком `locations[...]`; отдельный файл `logs/locations_<job_id>.csv` содержит сводку по каждой локации (включая `persisted_location_id`, длительность, список клипов, centroid).
- CLI и GUI показывают количество локаций, статистику по длительности/клипам и числа persisted/matched/new.
- Флаг `--export-locations-only` (и чекбокс «Export locations only») выполняет анализ и экспорт артефактов без взаимодействия с Resolve (без SetClipColor и маркеров).
- Флаг `--add-resolve-markers` (чекбокс «Add markers») добавляет маркеры начала/конца каждой локации с цветом кластера; undo-файл сохраняет координаты маркеров и удаляет их при `--undo` или `--undo-markers`.
- Хранилище локаций расположено в каталоге `state/`: файл `locations.sqlite` и папка `locations/` для JSON-режима.

## Live sensitivity tuning
- Слайдер «Sensitivity» (0⁠–⁠100, дефолт 60) управляет сразу двумя параметрами локаций: `similarity_threshold = 0.80 + 0.19 * (S/100)` и `min_duration_sec = round(1.0 + 4.0 * (1 - S/100), 1)`. В подписи рядом UI показывает вычисленные значения (`sim=…`, `min_dur=…`).
- Предустановки «Loose»/«Default»/«Strict» выставляют S=35/60/85. Пример: `S=30 → sim≈0.857, min_dur≈3.8 s` (агрессивное слияние), `S=85 → sim≈0.962, min_dur≈1.6 s` (разделяет чаще).
- Если пользователь меняет «Advanced» спинбоксы Min duration/Similarity, слайдер остаётся на месте, но перед запуском именно эти ручные значения имеют приоритет (UI выводит статус «Advanced»).
- Кнопка «Clear log» очищает панель логов; Accept/Reject фиксируют операторскую оценку последнего run-id.
- Каждый запуск добавляет строку в `.cache/tuning_log.jsonl` с настройками (S, similarity, min_dur, hysteresis, clips, clusters, cache_hits, quality_metric). Accept/Reject записывают отдельную строку с `accepted=true|false`. История автоматически ограничена последними 2000 строками.
- В summary и GUI дополнительно отображается «quality metric»: если `k>1`, это silhouette; иначе рассчитывается `median_hue_spread`, что помогает видеть, насколько материал однороден. Эти же поля попадают в JSON/CSV отчёты.

## Auto mode
- CLI: `--auto-mode` включает адаптивный подбор `similarity_threshold` и `min_duration_sec`. Дополнительно можно задавать `--auto-target-k` (минимальное число кластеров, по умолчанию 2) и `--auto-max-iters` (по умолчанию 3). При автоподборе каждое повторение печатается в логе как `Auto attempt: …`.
- GUI: чекбокс «Auto» блокирует ручной слайдер/спинбоксы и показывает выбранные параметры по завершении анализа. История доступна в логах, а кнопка Accept сохраняет профиль в `state/auto_profiles.json` (по связке project_id + clip_context_hash).
- Алгоритм измеряет silhouette (если `k>1`) или медианный разброс оттенков; при слабой раздельности повторяет анализ, ужесточая пороги (similarity→0.99, min_duration→1.0). При достижении условий (`k≥target_k` и достаточный прирост silhouette) останавливается.
- Отчёты включают `auto_mode`, `auto_iterations`, `auto_similarity_threshold`, `auto_min_duration_sec`, историю попыток (`auto_history`) и идентификаторы проекта/клипов (`project_id`, `clip_context_hash`) для обучения на будущих Accept.

## Resolve integration & exports
- `--add-resolve-markers` создаёт в Resolve по два маркера (start/end) на каждый `location_id`, название `Loc <id>` и цвет из палитры кластера. В GUI за это отвечает чекбокс «Add markers».
- Undo-файл (`undo/<run_id>.json`) теперь содержит секцию `markers`; команда `--undo <run_id>` восстанавливает цвета и удаляет маркеры, а `--undo-markers <run_id>` удаляет только маркеры.
- `--report-format json|csv|both` управляет серверными артефактами: можно писать только JSON, только CSV или оба формата (по умолчанию both). GUI предоставляет выпадающий список «Report».
- CSV по клипам дополнен колонками `location_color_name`, `persisted_location_id`; сводный location-CSV включает `color_name` и список clip_ids.
- GUI показывает ошибки запуска в критическом диалоге и включает все параметры локаций (Min duration, Similarity, Hysteresis) плюс формат отчётов.

## Sampling & Features
- Бэкенд сэмплинга задаётся через флаг `--sampler-backend auto|resolve|ffmpeg` (по умолчанию `auto`: сначала пробует Resolve, затем ffmpeg).
- Частота сэмплов регулируется `--max-frames-per-clip` (по умолчанию 5) и `--min-spacing-sec` (по умолчанию 2.0 сек между кадрами).
- Пример запуска: `python3 scripts/pigmentbucket_run.py --dry-run --limit 3 --sampler-backend ffmpeg --max-frames-per-clip 3 --min-spacing-sec 3 --max-k 6`.
- ffmpeg создаёт временные PNG-файлы в `/tmp`; при тестировании больших таймлайнов рекомендуется использовать `--limit` и уменьшать число кадров.
- Для каждого клипа вычисляются средние/медианные RGB/HSV характеристики и гистограммы (8 бинов на канал), итоговый вектор появляется в отчёте (`features`).
- Повторные запуски используют файловый кэш `.cache/features/<hash>.npz`; отключение через `--ignore-cache` (CLI) или чекбокс «Ignore cache» в GUI.

## Stop & Undo
- Каждый запуск получает `run_id` (печатается после отправки задания). Undo-файл можно найти в каталоге `undo/<run_id>.json`.
- Если `summary.coloring` сообщает `skipped`, цвета не затрагиваются и undo-артефакт не создаётся (например, при `k==1`).
- Флаг CLI `--force-min-k 2` и чекбокс GUI «Force min clusters» принудительно делят материал на два кластера (best-effort), если auto-K вернул `k==1`.
- Откат таймлайна: `python3 scripts/pigmentbucket_run.py --undo <run_id>` (восстанавливаются цвета клипов и удаляются маркеры локаций, если они были добавлены). После undo статус job меняется на `undone`.
- Только удалить маркеры (без изменения цветов): `python3 scripts/pigmentbucket_run.py --undo-markers <run_id>`.
- Принудительная остановка: `python3 scripts/pigmentbucket_run.py --stop-job <job_id>` либо `Ctrl+C` во время ожидания — раннер отправит `/jobs/<id>/stop`.
- GUI: кнопка «Undo Last» появляется после успешного запуска (снимок создаётся автоматически), checkbox «Dry run» переключает режим, кнопка Stop посылает terminate процессу.
- HTTP таймаут по умолчанию 60 с; меняется через `--http-timeout`, `PIGMENT_HTTP_TIMEOUT` или спинбокс в GUI.

## Jobs & History
- SQLite база лежит в `data/pigmentbucket.db`; создаётся автоматически при первом запуске сервиса.
- Быстрый просмотр: `curl "http://127.0.0.1:8765/jobs?limit=10"`
- Детали по конкретному job: `curl "http://127.0.0.1:8765/jobs/<job_id>"`
- CLI-хелперы: `python3 scripts/pigmentbucket_run.py --list-jobs --jobs-limit 5` и `python3 scripts/pigmentbucket_run.py --job <job_id>`
- Ротация логов и ретраи настраиваются через переменные окружения: `PIGMENT_MAX_LOG_FILES`, `PIGMENT_MAX_LOG_BYTES`, `PIGMENT_ANALYZE_MAX_RETRIES`, `PIGMENT_ANALYZE_RETRY_DELAY_MS`, `PIGMENT_ANALYZE_JOB_TIMEOUT_MS`.

## GUI
- Запуск PyQt-оболочки: `python3 scripts/pigmentbucket_gui.py` (PyQt6 6.6.1 / Qt 6.6.1 — см. раздел Troubleshooting).
- Панель управления включает чекбоксы `Dry run`, `Force min clusters`, `Ignore cache`, `Export locations only`, `Add markers`, комбобоксы Store (`sqlite`/`json`), Report (`both/json/csv`), слайдер Sensitivity с пресетами `Loose/Default/Strict`, кнопки Accept/Reject, Clear log и спинбоксы `HTTP timeout`, `Min duration`, `Similarity`, плюс чекбокс `Hysteresis`.
- Кнопка Analyze вызывает CLI с передачей всех параметров (force-min-k, ignore-cache, min-duration, similarity, hysteresis, locations-store, export-only, markers).
- Stop корректно завершает поток воркера; `PIGMENT_UI_FORCE_MOCK=1` оставляет моковый режим.
- Лог-окно показывает подробные сводки по локациям (persisted/matched/new, store backend), quality metric и предупреждения о гомогенном материале; записи стримятся в `.cache/gui_stream.jsonl` c автоматическим усечением до 2000 строк.

## Troubleshooting
- PyQt диалог сообщит, если установленная версия не совпадает с зафиксированной (6.6.1). Перезапустите окружение и выполните `python3 -m pip install -r requirements.txt --force-reinstall`.

## Заметки
- По умолчанию анализируется весь активный таймлайн.
- Флаг --limit N в раннере позволяет быстро тестировать на первых N клипах.
- Для изоляции зависимости запускайте через .venv.
