---
description: Запускает одну итерацию параллельного dental-dataset scan (9 search + 1 summary)
argument-hint: [optional: focus-hint]
---
Запусти ОДНУ итерацию dental-dataset scanner:

1. Прочитай `.claude/dental-scan/config.json`. Если файла нет — создай с `iteration: 1, exclude_ids: [], stop_recommended: false`.
2. Если `stop_recommended: true` и пользователь не передал `--force` в $ARGUMENTS — выведи master_manifest stats и завершись.
3. Создай папку `.claude/dental-scan/iter_{N}/`.
4. Прочитай поле `active_agents` в config. Если отсутствует — считай, что активны все 9 search-агентов. **ПАРАЛЛЕЛЬНО** (в одном turn-е, одним сообщением с несколькими Task tool invocations подряд) запусти только агентов из `active_agents`:
   - dental-search-caries
   - dental-search-perio
   - dental-search-noncarious
   - dental-search-soft-tissue
   - dental-search-restorative
   - dental-search-ortho-pediatric
   - dental-search-view
   - dental-search-ru-cis
   - dental-search-asia-latam-africa
   - dental-search-surgical (новый: флегмоны, MRONJ, фистулы, оголённые корни, dry socket, gingival overgrowth)
   - dental-search-outreach (новый: не ищет датасеты, готовит CSV-трекер и email-шаблоны по request-only leads; запускать последним)

   Каждому search-агенту передай как prompt: `"Iteration: {N}. Output file: agent_XX_<name>.jsonl. Focus hint: $ARGUMENTS. Read config at .claude/dental-scan/config.json before starting."`
5. Дождись всех активных search-агентов.
6. Запусти dental-search-summary.
7. Опционально запусти dental-search-outreach если он в active_agents (после summary, он читает master_manifest).
7. Выведи пользователю итоговый REPORT.md.

**Важно:** запускай все 9 search-агентов ОДНОВРЕМЕННО в одном ответе (один message, много параллельных Task calls). НЕ последовательно.
