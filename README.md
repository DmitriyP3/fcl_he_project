# Федеративное непрерывное обучение с гомоморфным шифрованием

Экспериментальная установка к магистерской диссертации «Разработка и
исследование методов применения гомоморфного шифрования для защиты
медицинских данных в цифровых системах» (Satbayev University, 7M06101).

Три клиники по очереди осваивают новые диагностические задачи, обучая одну
общую модель. Снимки не покидают клиники; общий пул подсказок L2P (11 520
чисел) шифруется CKKS, сервер складывает шифртексты без секретного ключа.

## Структура

```
core/                    ядро (только NumPy + TenSEAL)
  config.py              все параметры: пул, обучение, CKKS, DP-SGD, протокол
  data.py                признаки -> клиники -> последовательные задачи
  pool.py                пул подсказок L2P и модель клиники, шаг DP-SGD
  crypto.py              агрегация: открытая или CKKS (сервер без секретного ключа)
  federation.py          главный цикл: задачи -> раунды -> оценка
  metrics.py             итоговая точность, забывание, нормализация обновлений
  attack.py              атака определения членства
  leakage.py             утечка расписания задач по последовательности обновлений
  dp_accountant.py       учёт приватности (ε, δ) для DP-SGD
run_experiment.py        все эксперименты (сценарии ниже)
compare_temporal.py      утечка задачи: основной прогон против контроля
comm_compare.py          реальные размеры шифртекстов: пул / вся модель / признаки
system.py                система: обучение, сохранение модели, диагноз по снимку
scripts/prepare_features.py   признаки ViT-Tiny из снимков MedMNIST
tests/                   автоматические тесты (pytest)
.github/workflows/ci.yml непрерывная интеграция (GitHub Actions)
```

## Установка

Требования: Python 3.12, Git. Проверено: Windows 11 (полный прогон),
Ubuntu (тесты в CI).

```bash
git clone <адрес репозитория>
cd <папка репозитория>
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Для видеокарты torch ставится по инструкции pytorch.org.

## Воспроизведение результатов

Данные и результаты в репозиторий не входят — они воспроизводятся командами.

1. Признаки из снимков MedMNIST (скачиваются автоматически):

```bash
python scripts/prepare_features.py --federation primary --mode real
python scripts/prepare_features.py --federation external --mode real   # для сценария external
```

2. Эксперименты:

```bash
python run_experiment.py --scenario crypto --protocol cil --seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
python run_experiment.py --scenario sensitivity --protocol cil --seeds 1 2 3 4 5
python run_experiment.py --scenario attack --seeds 1 2 3 4 5 --targets 400
python run_experiment.py --scenario temporal --protocol cil --seeds 1 2 3 --out results/temporal_cil
python run_experiment.py --scenario temporal --protocol cil --control --seeds 1 2 3 --out results/temporal_cil_control
python compare_temporal.py --main results/temporal_cil/temporal_observations.npz --control results/temporal_cil_control/temporal_observations.npz
python comm_compare.py
```

Результаты сохраняются в `results/` в формате JSON вместе с параметрами
запуска, поэтому любой результат можно повторить той же командой.

| Сценарий | Вопрос |
|---|---|
| `crypto` | меняет ли CKKS точность и забывание? |
| `sensitivity` | во сколько раз нужно усилить погрешность CKKS, чтобы эффект появился? |
| `attack` | что на самом деле защищает от атаки определения членства? |
| `temporal` | узнают ли сервер и клиника-участник по последовательности обновлений, какие диагнозы изучает клиника? |
| `temporal --control` + `compare_temporal.py` | отделяет утечку задачи от естественного дрейфа обучения |
| `dpsgd` | формальная гарантия (ε, δ) для каждого пациента и её цена в точности |
| `aggregation` | помогает ли взвешивание по использованию ячеек пула? |
| `defense` | портит ли нормализация обновлений точность? |
| `external` | воспроизводятся ли выводы на второй федерации? |
| `single` | один прогон для отладки |

Ключевые флаги: `--protocol til|cil` (номер задачи известен / неизвестен),
`--crypto plaintext|ckks`, `--defense none|global|percomponent|dp`,
`--seeds`, `--tasks`, `--rounds`, `--out`.

3. Система «снимок → диагноз»:

```bash
python system.py train                                   # 3 клиники, CKKS + DP-SGD -> models/fcl_system.npz
python system.py selftest                                # путь предсказания совпадает с путём обучения
python system.py predict --clinic blood --image cell.jpg # диагноз по снимку
```

## Тесты и непрерывная интеграция

```bash
pip install -r requirements-ci.txt
pytest -v
```

Тесты в папке `tests/` (14 штук):

- `test_metrics.py` — итоговая точность, забывание и нормализация обновлений;
- `test_crypto.py` — шифрованное усреднение совпадает с открытым, у сервера нет секретного ключа, пул занимает 3 шифртекста;
- `test_dp_accountant.py` — счётчик приватности совпадает с эталонными значениями;
- `test_pool.py` — выбор ячеек пула и регрессионный тест шага DP-SGD;
- `test_integration.py` — сквозной прогон: признаки → обучение с CKKS → файл результатов.

CI (GitHub Actions, `.github/workflows/ci.yml`) запускает все тесты при каждой
отправке кода в ветку `main` и при каждом pull request.
