# Порядок работы

1. **Baseline-train:** сначала проверяем, что обучение стабильно и даёт адекватный `val_acc`

2. **Baseline-landscape:** берём пару независимых минимумов в одном режиме и измеряем интерполяцию/барьер/геометрию. Это "нулевая точка"

3. **Noise grid:** систематически меняем режим шума (`batch`/`lr`), повторяем весь цикл, собираем таблицы, сравниваем

4. **(Опц) Polyline paths:** если линейный путь даёт барьер, проверяем, снижается ли он через pivot(ы)


---

# Эксперимент 1 — Baseline обучение

## Задача

Получить устойчивые чекпоинты минимумов и опорное качество **ResNet18 на CIFAR-10**

## Конфигурация

`configs/train/sgd_base.yaml`  

Количество `seed`: **5**

## Запуски

Для каждого seed $s$:
ntempvh train --config configs/train/sgd_base.yaml --seed s --out outputs/runs


## Минимальные sanity checks

- **val_acc** заметно больше `0.1` (иначе это "угадывание" CIFAR-10)
- В логах нет `NaN/inf` и нет явной дивергенции loss
- Между seed есть **разброс**, но не хаос: качество в одном диапазоне


---

# Эксперимент 2 — Linear interpolation и barrier (mode connectivity proxy)

## Что проверяем

Насколько **соединены два независимых минимума**, найденные одинаковым режимом обучения.

Операционально: строим $L(t)$ вдоль линейного пути и измеряем $\Delta L$.

## Выбор пары минимумов

Берём два run с одинаковыми гиперпараметрами и разными seed:

$$
A=\texttt{.../seed s_0/.../final.pt}, \qquad
B=\texttt{.../seed s_1/.../final.pt}.
$$

## Интерполяция

Конфиг: `configs/eval/interpolation.yaml`

Команда:
ntempvh interpolate --ckptA A --ckptB B --config configs/eval/interpolation.yaml


## Barrier

Конфиг: `configs/eval/barrier.yaml`

Команда:
ntempvh barrier --interp_csv outputs/artifacts/interp/<interp.csv>
--config configs/eval/barrier.yaml


## Как интерпретируем

- Большой $\Delta L$ — выраженный **"барьер"** по выбранному пути  
- Малый $\Delta L$ — практически **no barrier** для линейной интерполяции  
- Смотрим не только $\Delta L$, но и $\Delta L_{\mathrm{rel}}$


---

# Эксперимент 3 — Geometry (локальная кривизна около минимума)

## Что проверяем

Прокси-показатель **"плоскости / кривизны" минимума** через центральную разность второго порядка по случайным направлениям.

## Запуск

Конфиг: `configs/eval/geometry.yaml`

Поля:

- `alpha` задаёт $\varepsilon=\alpha\|\theta\|$
- `num_directions` — точность оценки
- `eval_batch_size` — ускорение и стабильность оценки

Для чекпоинтов $A$ и $B$:
ntempvh geometry --ckpt A --config configs/eval/geometry.yaml
ntempvh geometry --ckpt B --config configs/eval/geometry.yaml


## Как интерпретируем

- Меньшее $\kappa_{\mathrm{tr}}$ → более **"плоский" минимум**
- Значение зависит от `alpha` и от данных, на которых оценивается loss (`val`)


---

# Эксперимент 4 — Noise grid (batch/lr) + полный цикл

## Идея

Меняем режим **шума SGD** и проверяем, как это отражается одновременно на:

качество ↔ барьеры ↔ геометрия.

## Сетка параметров

Минимум:

$$
\mathrm{batch}\in\{64,128,256\},\quad
\mathrm{lr}\in\{0.05,0.1,0.2\},\quad
\mathrm{seed}\in\{0,1\}.
$$

## Протокол для одной точки сетки $(\mathrm{batch},\mathrm{lr})$

1. Два независимых обучения (`seed 0` и `1`) → два `final.pt`
2. Интерполяция между ними → `interp.csv`
3. Barrier по `interp.csv` → `barrier.json` и строка в `barriers.csv`
4. Geometry хотя бы для одного чекпоинта → `geometry.json` и строка в `geometries.csv`

## Что в итоге сравниваем

Для каждого режима:

- финальный `val_acc` (и/или лучший по `val_loss`)
- $\Delta L$ и $\Delta L_{\mathrm{rel}}$
- $\kappa_{\mathrm{tr}}$

Дальше строим зависимости:

- $\Delta L$ vs `(batch, lr)`
- $\kappa$ vs `(batch, lr)`
- `val_acc` vs $\Delta L$
- `val_acc` vs $\kappa$


---

# Эксперимент 5 — Piecewise-linear (pivot) пути

## Зачем

Если линейный путь показывает барьер, проверяем, что барьер **можно снизить** полилинией через pivot.

Это демонстрация того, что "барьер" может быть **артефактом выбранного пути**.

## Настройки

В `configs/eval/interpolation.yaml`:
path:
type: piecewise_linear
num_points: 41
bn_recalib_batches: 20
pivots:
- /path/to/pivot1.pt


## Запуск
ntempvh interpolate --ckptA A --ckptB B --config configs/eval/interpolation.yaml

ntempvh barrier --interp_csv outputs/artifacts/interp/<interp_piecewise.csv>
--config configs/eval/barrier.yaml


## Критерий

Сравниваем:

$$
\Delta L_{\text{linear}} \quad \text{vs} \quad \Delta L_{\text{piecewise}}
$$

на одинаковой паре $(A,B)$
