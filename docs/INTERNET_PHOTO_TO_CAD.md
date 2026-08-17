# Фотографии -> CAD

Текущий основной маршрут DA3-CAD принимает несколько согласованных ракурсов
одного неподвижного предмета и возвращает редактируемый CadQuery/STEP-кандидат
вместе с честным решением `ACCEPT` или `ABSTAIN`.

Это не восстановление исходного feature tree и не автоматическая метрология.
Без известного физического размера единицы результата остаются каноническими.

## Контракт входа

Нужны 30-60 резких фотографий одного физического экземпляра:

- предмет и фон неподвижны, перемещается камера;
- соседние кадры имеют 60-80% перекрытия;
- есть нижний, средний и верхний пояса ракурсов;
- присутствуют противоположные стороны, а не только экваториальная дуга;
- фокус, зум и экспозиция по возможности зафиксированы;
- для каждого кадра задана бинарная маска выбранного экземпляра.

Пользователь знает, какой предмет он снимает, но не обязан указывать его класс
или CAD-тип. Маска выбирает физический экземпляр; словарь деталей не
используется.

Случайные фотографии похожих товаров из интернета обычно не подходят:
экземпляр, оптика, масштаб и скрытая геометрия могут отличаться. Нужен
многовидовой набор одного предмета.

## Алгоритм

```text
несколько RGB + маски выбранного экземпляра
        |
exhaustive COLMAP SfM -> калиброванные камеры
        |
точное undistort RGB и тех же масок
        |
masked CUDA PatchMatch
        |
cross-view-confirmed raw fused cloud
        |                         \
измеренные revolve/sketch roots    Poisson render -> CADENA proposals
        \                         /
конкурирующие CadQuery-кандидаты
        |
проекция обратно во все исходные виды
        |
silhouette + depth + edge gates
        |
OpenCascade: один valid solid -> STEP candidate
        |
ACCEPT или ABSTAIN
```

Ключевой контракт: сырое слитое облако является входом измерительного
CAD-фиттера. Poisson-поверхность менее плотная и может быть не watertight; она
используется как стабильный render для proposer, но не заменяет исходные точки.

Прямые корневые гипотезы не являются классами деталей:

- `revolve`: произвольный измеренный осевой профиль и полный оборот;
- `sketch-extrusion`: произвольный замкнутый профиль из линий/окружности и
  extrusion вдоль лучшей оси;
- restricted CADENA: дополнительный learned-кандидат.

Все кандидаты проходят одинаковые kernel/source-view gates. У pooled cloud нет
per-view identity, поэтому прямой revolve не имеет права придумывать shell или
внутреннюю стенку. Такая топология требует отдельного многовидового
доказательства.

## Запуск

Сначала восстанавливаются камеры по полным исходным кадрам. Для
неупорядоченных фотографий нужен exhaustive matching. Переданные source masks
undistort-ятся той же моделью камеры, что и RGB:

```bash
da3-cad prepare-photos-sfm photos/ \
  --masks source_masks/ \
  --output work/sfm \
  --pairing exhaustive
```

Если source masks ещё нет, `--masks` можно опустить. После SfM создайте masks
или SAM2 boxes непосредственно для `registered_frames/`; в следующей команде
передайте соответствующий `--masks` или `--boxes`.

Затем объект вырезается уже в зарегистрированных undistorted кадрах:

```bash
da3-cad prepare-target work/sfm/registered_frames \
  --masks work/sfm/registered_masks \
  --cameras work/sfm/cameras.npz \
  --output work/target
```

Строится измеренная геометрия:

```bash
da3-cad dense-surface work/target/images \
  --masks work/target/masks \
  --cameras work/target/cameras.npz \
  --output work/dense \
  --mvs-python .venv-mvs/bin/python
```

И затем CAD:

```bash
da3-cad fit-cad work/dense/surface.ply \
  --measurements work/dense/fused_cloud.ply \
  --output work/cad \
  --cadena-checkout data/upstream/cadena \
  --cadena-checkpoint data/checkpoints/cadena/rl \
  --verification-workspace work/dense/mvs \
  --cameras work/target/cameras.npz
```

Код возврата `0` означает, что `model.step` прошёл kernel и source-view
гейты. Код `3` означает `ABSTAIN`: сохраняются `candidate.py`,
`candidate.step`, preview и причины отказа.

## Что проверяется

До CAD-фиттинга виды без достаточной измеренной глубины исключаются из
верификации: требуется минимум 128 depth pixels и покрытие минимум 10% target
mask. Это не ослабляет пороги кандидата; плохой depth-view просто не считается
измерением.

Финальный кандидат должен пройти:

- mean silhouette IoU не ниже 0.87;
- depth inlier fraction при 3% не ниже 0.90;
- при наличии внутренних RGB-границ: edge precision не ниже 0.45 и recall не
  ниже 0.12;
- ровно один положительный OpenCascade solid;
- `isValid() == true` и непустой STEP export.

Валидный B-Rep может быть геометрически неверным. Поэтому kernel-valid STEP без
source-view соответствия остаётся `candidate.step`, а не успешным
`model.step`.

Линии и треугольники в STL-preview не являются CAD-топологией. Авторитетный
результат -- STEP и его kernel report.

## Роль DA3

DA3 больше не является источником камер или основной метрической геометрии.
В dense-маршруте камеры даёт COLMAP, а глубину -- calibrated PatchMatch.

DA3 сохраняется как экспериментальный confidence-aware prior для sparse или
textureless 2DGS. Он выключен по умолчанию, пока held-out multi-seed тесты не
покажут устойчивый выигрыш. Подробности:
[BREPGAUSSIAN_DA3.md](BREPGAUSSIAN_DA3.md).

## Проверенный real-RGB аудит

Пять физических T-LESS объектов прогнаны по 32 RGB-кадрам каждый:

| Case | Выбранный root | Direct IoU | Решение |
|---|---|---:|---:|
| o02-fixed | measured revolve | 0.310 | **ABSTAIN** |
| o04-fixed | measured revolve | 0.368 | **ABSTAIN** |
| o10 | measured sketch-extrusion | 0.162 | **ABSTAIN** |
| o20-fixed | measured sketch-extrusion | 0.398 | **ABSTAIN** |
| o25 | restricted CADENA | 0.644 | **ABSTAIN** |

Все пять STEP-кандидатов kernel-valid и содержат один solid, но каждый нарушает
хотя бы один frozen source-view gate. Средний no-alignment IoU вырос с 0.3263
до 0.3763; это инженерный прогресс, а не готовое универсальное решение и не
SOTA.

BOP masks и fixed instance index являются disclosed target-selection oracles.
Reference CAD открывался только post-hoc. Ledger:
[results/real-photo-e2e-v1.json](results/real-photo-e2e-v1.json). Локальный
семистраничный отчёт строится командой:

```bash
python scripts/build_real_photo_e2e_report.py
```

T-LESS RGB и производные изображения не распространяются репозиторием.

## Текущие ограничения

Нельзя обещать:

- исходное дерево операций, design intent, допуски, GD&T и посадки;
- миллиметры без известного размера или внешней метрической калибровки;
- скрытые полости и обратную сторону при отсутствии наблюдений;
- надёжные резьбы, мелкие фаски/скругления и элементы ниже stereo resolution;
- прозрачные, зеркальные, очень тонкие или движущиеся детали;
- assemblies, joints и несколько независимо движущихся тел;
- universal photo-to-CAD или SOTA по пяти объектам.

RTX 5080 16 GB достаточна для текущего inference. H100 ускоряет большие
эксперименты, но не исправляет неполные ракурсы или неверную грамматику.
