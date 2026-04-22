---
description: Запускает одну итерацию параллельного dental-dataset scan (9 search + 1 summary)
argument-hint: [optional: focus-hint]
---
Запусти ОДНУ итерацию dental-dataset scanner:

1. Прочитай `.claude/dental-scan/config.json`. Если файла нет — создай с `iteration: 1, exclude_ids: [], stop_recommended: false`.
2. Если `stop_recommended: true` и пользователь не передал `--force` в $ARGUMENTS — выведи master_manifest stats и завершись.
3. Создай папку `.claude/dental-scan/iter_{N}/`.
4. **ПАРАЛЛЕЛЬНО** (в одном turn-е, одним сообщением с несколькими Task tool invocations подряд) запусти 9 субагентов:
   - dental-search-caries
   - dental-search-perio
   - dental-search-noncarious
   - dental-search-soft-tissue
   - dental-search-restorative
   - dental-search-ortho-pediatric
   - dental-search-view
   - dental-search-ru-cis
   - dental-search-asia-latam-africa

   Каждому передай как prompt: `"Iteration: {N}. Output file: agent_XX_<name>.jsonl. Focus hint: $ARGUMENTS. Read config at .claude/dental-scan/config.json before starting."`
5. Дождись всех 9.
6. Запусти dental-search-summary.
7. Выведи пользователю итоговый REPORT.md.

**Важно:** запускай все 9 search-агентов ОДНОВРЕМЕННО в одном ответе (один message, много параллельных Task calls). НЕ последовательно.
