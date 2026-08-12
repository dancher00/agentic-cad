# Интернет-фото и видео → CAD без Cadrille

Текущий основной маршрут DA3-CAD не использует Cadrille:

```text
RGB-кадры
  → DA3-LARGE: depth + confidence + K/E
  → автоматическая маска объекта
  → многовидовой visual hull с depth carving
  → связный voxel-объём
  → детерминированное разложение на cuboid features
  → CadQuery → проверенный STEP + STL
```

Это уже рабочий путь до валидного B-Rep. Он восстанавливает грубую внешнюю форму,
но не исходное дерево построения детали.

## Какой вход подходит

Лучший вход — 12–24 разных ракурса одного и того же неподвижного объекта:

- объект находится примерно в центре и целиком виден;
- соседние ракурсы перекрываются;
- меняется камера, а не геометрия объекта;
- фиксированы зум и фокусное расстояние;
- нет сильного motion blur и бликов;
- желательно видеть верх, низ и боковые стороны.

Набор случайных изображений «похожей модели» из разных карточек товара не
эквивалентен многовидовой съёмке: комплектация, объектив, подвижные части и
геометрия могут различаться. Такой набор нужно сначала отфильтровать до одного
физически согласованного экземпляра.

## Самый простой запуск: только RGB

```bash
./.venv/bin/da3-cad reconstruct photos/ \
  -o outputs/object-rgb-only \
  --config configs/internet_photo.yaml \
  --accept-noncommercial-weights
```

Здесь DA3 оценивает и глубину, и камеры. Маски строятся автоматически. Ни
`--cameras`, ни `--masks`, ни отдельные CAD-веса не нужны.

Если известна реальная ширина, её можно передать сразу:

```bash
./.venv/bin/da3-cad reconstruct photos/ \
  -o outputs/object-mm \
  --config configs/internet_photo.yaml \
  --known-dimension body_width=120mm \
  --accept-noncommercial-weights
```

Без такого измерения выход остаётся в явно помеченных canonical model units.

## Более надёжный запуск из видео

Сначала извлекаются разнообразные кадры и автоматически восстанавливаются
камеры COLMAP:

```bash
./.venv/bin/da3-cad prepare-video object.mp4 \
  -o captures/object --views 24

./.venv/bin/da3-cad reconstruct \
  captures/object/colmap/registered_frames \
  -o outputs/object-colmap \
  --config configs/internet_photo.yaml \
  --cameras captures/object/colmap/cameras.npz \
  --accept-noncommercial-weights
```

Если автоматические маски ошибаются, можно передать бинарные PNG с совпадающими
именами и использовать `configs/internet_photo_masked.yaml`.

## Что делает алгоритм после RGB

1. DA3-LARGE выдаёт per-view `depth`, `confidence`, intrinsics `K` и
   world-to-camera extrinsics `E`.
2. Сегментатор `internet-object-depth-seeded-grabcut-v2` ищет ближний связный
   компонент в маленькой центральной области DA3-depth. GrabCut может расширить
   маску только локально вокруг этого компонента, поэтому фон не захватывается
   целиком.
3. Каждый выбранный пиксель разворачивается в 3D:
   `x_cam = z K⁻¹[u,v,1]ᵀ`, затем `x_world = E⁻¹x_cam`.
4. Точки всех видов фильтруются по confidence, объединяются и ориентируются в
   канонической системе.
5. Внутри robust bbox строится сетка 16 voxel по длинной оси. Voxel остаётся,
   если проходит маски в достаточной доле видимых камер; DA3 depth отсекает
   пространство перед наблюдаемой поверхностью.
6. Сохраняется крупнейшая 6-связная компонента. Edge/vertex pinch-паттерны
   минимально заполняются, чтобы граница оставалась manifold.
7. Greedy-разложение покрывает voxel-объём непересекающимися cuboid features.
   Из них генерируется редактируемый CadQuery с параметрами
   `body_width/body_depth/body_height`.
8. Код выполняется в ограниченном subprocess. Экспорт разрешён только для
   одного валидного solid с конечным положительным объёмом.
9. Готовая модель независимо сравнивается с входным облаком и со всеми
   входными масками. Результат пишется в
   `artefacts/input_fit_validation.json`; ground-truth CAD не используется.

## Проверенный интернет-пример

Источник: 24 кадра видео камеры Google Objectron
(`captures/internet_objectron_camera/source.json`, C-UDA-1.0). Это один
интеграционный пример, а не benchmark точности по категориям.

| Вход | Камеры | Маски | STEP/STL | Cuboids | Input CD² | Trimmed silhouette IoU |
|---|---|---|---|---:|---:|---:|
| 24 RGB | DA3 | автоматические v2 | 1 watertight solid | 69 | 0.02641 | 82.30% |
| 24 RGB | COLMAP из того же видео | автоматические v2 | 1 watertight solid | 44 | 0.02425 | 85.27% |

Silhouette IoU здесь измеряет согласие с теми масками, которые были входом
реконструкции. Это полезная self-consistency проверка, но не IoU с неизвестным
эталонным CAD. Для результата с COLMAP также сохранены все 24 проекции; средний
IoU равен 84.83%.

Готовые результаты:

- `outputs/internet_objectron_camera/reconstruction_visual_hull_da3_only/`;
- `outputs/internet_objectron_camera/reconstruction_visual_hull_auto_v2/`.

## Что находится на выходе

- `model.py` — исполнимый и редактируемый CadQuery;
- `model.step` — один проверенный B-Rep solid;
- `model.stl` — герметичная triangulation;
- `parameters.json` — единицы, scale channel и параметры;
- `quality.json`, `report.md`, `provenance.json`;
- depth/confidence/mask overlays, fused cloud, canonicalizer trace;
- GT-free Chamfer и multiview silhouette validation.

## Чего пока нет

- восстановленного design intent и исходной истории операций;
- точных отверстий, фасок, радиусов и карманов для произвольной детали;
- резьб, допусков, посадок, GD&T и сборок;
- метрического масштаба без калибровки или одного известного размера;
- гарантии правильной скрытой геометрии из одного изображения;
- benchmark-доказательства точности на большом наборе интернет-объектов.

RTX 5080 16 ГБ достаточно для этого inference-конвейера: на проверенном
24-кадровом примере forward DA3-LARGE занимал около одной секунды после загрузки
модели. H100 нужен дальше для обучения feature-recognition/primitive fitting и
для большого benchmark, а не для запуска текущего visual-hull backend.
