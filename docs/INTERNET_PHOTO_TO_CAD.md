# Интернет-фото → CAD

Основной маршрут DA3-CAD принимает несколько ракурсов одного неподвижного
предмета и выдаёт редактируемый CadQuery, один проверенный STEP solid, STL,
параметры и полный provenance.

```text
RGB + выбранный target (box или mask)
    → SAM2 mask → общий target/context crop
    → DA3-LARGE-1.1 (depth/confidence/K/E)
    → observed masked depth + trusted fusion + filtered surface
    → camera coverage + missing-view suggestions
    → CAD grammar:
        line/circle sketch + cuts + extrude
        или axial profile + revolve
    → безопасная проверка CadQuery
    → model.py + STEP + STL + отчёты
```

## Как снимать

Оптимальный практический вход — 12–24 кадра:

- один и тот же физический объект, геометрия не меняется;
- объект целиком виден и находится примерно в центре;
- соседние ракурсы имеют большое перекрытие;
- камера обходит объект, зум и фокус фиксированы;
- есть боковые, верхние и несколько нижних наклонных видов;
- нет сильного motion blur, пересветов и движущихся частей.

Случайные фотографии похожей модели из разных магазинов не образуют корректную
многовидовую последовательность: экземпляры, объективы и скрытые варианты
геометрии могут различаться. Сначала нужно отобрать согласованный набор одного
объекта.

## Выбор target и запуск

Пользователь или робот знает, какой физический предмет выбран в кадре. Он не
обязан знать класс или CAD-тип предмета: достаточно loose bounding box либо
готовой маски для каждого вида. Сегментация поэтому выполняется до DA3.

```bash
da3-cad prepare-target photos/ \
  --boxes boxes.json \
  --output captures/object \
  --segment-device cuda

da3-cad doctor captures/object/images
da3-cad reconstruct captures/object/images \
  --output outputs/object \
  --config configs/internet_photo_masked.yaml \
  --masks captures/object/masks \
  --accept-noncommercial-weights
```

`boxes.json` хранит по одному `xyxy` для каждого имени изображения в
`normalized-exif-corrected-xyxy` или `pixel-exif-corrected-xyxy`. Это указание
экземпляра для SAM2, а не class label и не автоматический detector.

Если источник уже дал точные PNG-маски с теми же stem:

```bash
da3-cad prepare-target photos/ \
  --masks source_masks/ \
  --output captures/object \
  --selection-source user-mask
```

`prepare-target` сохраняет маску, выбирает одинаковый размер crop для всех видов,
оставляет реальный контекст и не растягивает отдельные изображения. При передаче
исходного `--cameras cameras.npz` intrinsics переводятся в координаты crop;

extrinsics не меняются.
## Камеры и масштаб

Внешние камеры обычно повышают устойчивость:

```bash
da3-cad reconstruct captures/object/images \
  --output outputs/object-calibrated \
  --config configs/internet_photo_masked.yaml \
  --masks captures/object/masks \
  --cameras captures/object/cameras.npz \
  --accept-noncommercial-weights
```

COLMAP и any-view DA3 не дают гарантированного физического масштаба. Чтобы
получить миллиметры, передайте один действительно измеренный именованный размер:

```bash
--known-dimension extrusion_length=120mm
```

Без такого свидетельства результат честно помечается как
`canonical-model-unit`.

## Что происходит после RGB

1. Пользователь или робот отмечает выбранный экземпляр box или mask.
2. SAM2 преобразует box в full-resolution mask; сомнительные виды уточняются
   детерминированными positive/negative prompts, holes сохраняются.
3. Один общий размер crop сохраняет projective scale между видами и контекст;
   маски переносятся в него без resize.
4. DA3-LARGE-1.1 выдаёт для каждого подготовленного вида z-depth, confidence,
   `K` и world-to-camera `E`.
5. Все finite positive depth-пиксели внутри target mask образуют observed channel;
   он остаётся плотной диагностикой и не называется очищенной геометрией.
6. Отдельный confidence gate образует trusted fusion для профиля и apertures;
   outlier/multi-view consistency строит filtered surface для ориентации и score.
7. Каноническая ориентация и scale channel вычисляются детерминированно; фильтр
   не считается более полным описанием объекта, чем raw evidence.
8. Ветка `extrude` проверяет ось, восстанавливает line/circle profile и принимает
   cut по повторяющейся дыре в masks либо по RGB-эллипсу, внутри которого DA3
   depth не объясняется локальной плоскостью. Raw 3D void остаётся предпочтительным
   измерением, но заполненная foreground-mask больше не стирает видимое отверстие.
9. Ветка `revolve` сначала проверяет raw 3D radial surface. После её отказа outer
   profile можно взять из нескольких согласованных silhouettes. Внутренний
   RGB-эллипс требует повторяемого концентрического обода и торцевых ракурсов.
   Обода распределяются между двумя концами силуэта, после чего сравниваются
   `solid`, две глухие полости и `through`; сквозной вариант разрешён только при
   evidence на обоих концах от достаточно разнесённых направлений камер.
10. Направления камер кластеризуются; при недостаточном покрытии отчёт предлагает
    недостающие направления, а одинаковые соседние video frames не считаются новыми видами.
11. CAD surface проецируется обратно в masks/depth и получает provenance
    measured, weakly measured, unobserved или contradicted. Достроение допустимо
    только при недостаточном покрытии и отсутствии видимого противоречия.
12. STEP принимается только как один solid с конечным положительным объёмом;
    иначе pipeline честно abstains без template или скрытого fallback.

## Пять реально проверенных интернет-объектов

Закреплены пять разрешённых Google Objectron videos. Для каждого извлекается
пул из 40 кадров: DA3 оценивает позы всего пула, mask-quality gate удаляет
испорченные сегментации, затем выбираются 24–40 разных ракурсов и выбранный
subset повторно проходит DA3 перед fusion/CAD.

| Объект | Кадры | Результат |
|---|---:|---|
| Книга | 40 → 24 | **ACCEPT**, `extrude`; measured 68,7%, contradicted 9,7% |
| Цилиндрическая бутылка | 40 → 40 | STEP candidate, **UNSAFE**; contradicted 42,4% |
| Камера | 40 → 40 | **ABSTAIN**: нужны составные body + lens |
| Кружка с ручкой | 40 → 40 | **ABSTAIN**: нужны shell + handle + union |
| Открытый ноутбук | 40 → 39 | **ABSTAIN**: пустой mask удалён; нужны две пластины и hinge |

Это integration gate, не benchmark точности: reference CAD и физического scale
нет. Из 200 pool-кадров 183 использованы в reconstruction pass. Два прогона
создали kernel-valid STEP, product gate принимает только книгу. Новые виды не
дали дополнительных ACCEPT, но убрали ложный coarse-extrude ноутбука и показали,
что pose diversity нельзя подменять surface provenance. Полный ledger:
[`results/real-photo-v3.json`](results/real-photo-v3.json).
Исторический visual-hull run остаётся в [`RESULTS.md`](RESULTS.md).

## Что пока нельзя обещать

- точную скрытую геометрию из одного фото;
- исходное дерево операций и design intent;
- надёжные отверстия, карманы, фаски, скругления и паттерны для любой детали;
- резьбы, допуски, посадки, GD&T, материалы и сборки;
- миллиметры без калибровки или известного размера;
- производственную точность без сравнения с reference CAD/метрологией.

RTX 5080 16 ГБ достаточна для текущего inference. H100 полезна для будущего
обучения распознавания CAD-features и больших benchmark-прогонов.
