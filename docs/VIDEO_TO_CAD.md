# Video to CAD

Поддерживаемая съёмка: один неподвижный жёсткий предмет, вокруг которого
движется RGB-камера. Видео сначала превращается в набор резких перекрывающихся
кадров; дальше используется тот же COLMAP/MVS/CAD путь, что и для фотографий.

## Как снимать

- сделайте медленный полный обход вместо быстрого pan;
- держите 60–80% overlap соседних кадров;
- добавьте верхние и нижние косые ракурсы;
- по возможности зафиксируйте zoom, focus, exposure и white balance;
- предмет, фон и освещение должны оставаться неподвижными;
- избегайте motion blur, рук, бликов и движущихся частей.

Обычный turntable не соответствует текущему SfM-контракту: камера и фон
неподвижны, а объект движется. Нужен moving-camera capture либо отдельный
object-centric pose estimator.

## 1. Извлечь кадры и камеры

```bash
da3-cad prepare-video object.mp4 \
  --output work/video \
  --views 32
```

`prepare-video` выбирает резкие и визуально/временно разнообразные кадры,
запускает sequential COLMAP, undistort и записывает provenance:

```text
work/video/
├── capture.json
├── frames/
└── colmap/
    ├── camera_recovery.json
    ├── cameras.npz
    └── registered_frames/
```

Для видео sequential pairing корректен, потому что кадр N перекрывается с
N+1. Для отдельной неупорядоченной фотосъёмки используйте
`prepare-photos-sfm --pairing exhaustive`.

## 2. Выбрать экземпляр

Создайте masks или boxes уже для `registered_frames/`. Пользователь знает,
какой физический экземпляр выбран; category label CAD-грамматике не нужен.

```bash
da3-cad prepare-target work/video/colmap/registered_frames \
  --masks registered_masks/ \
  --cameras work/video/colmap/cameras.npz \
  --output work/target
```

С boxes вместо masks можно включить SAM2:

```bash
da3-cad prepare-target work/video/colmap/registered_frames \
  --boxes boxes.json \
  --cameras work/video/colmap/cameras.npz \
  --output work/target \
  --segment-device cuda
```

## 3. Построить измеренную поверхность

```bash
da3-cad dense-surface work/target/images \
  --masks work/target/masks \
  --cameras work/target/cameras.npz \
  --output work/dense \
  --mvs-python .venv-mvs/bin/python
```

Авторитетное измерение — `work/dense/fused_cloud.ply`. Poisson
`surface.ply` нужен как conditioning render и может быть визуально менее
плотным.

## 4. Построить и проверить CAD

```bash
da3-cad fit-cad work/dense/surface.ply \
  --measurements work/dense/fused_cloud.ply \
  --output work/cad \
  --cadena-checkout data/upstream/cadena \
  --cadena-checkpoint data/checkpoints/cadena/rl \
  --verification-workspace work/dense/mvs \
  --cameras work/target/cameras.npz
```

Код возврата 0 означает accepted `model.step`. Код 3 означает `ABSTAIN`:
`candidate.step` и причины отказа сохраняются для аудита.

DA3 в этом dense video path не даёт ни камеры, ни основную глубину. Его
экспериментальная роль — confidence-aware prior для sparse/textureless 2DGS; по
умолчанию он выключен.

## Масштаб и типичные ошибки

COLMAP восстанавливает геометрию с неизвестным similarity scale. До добавления
известного размера CAD остаётся в canonical units.

- мало registered frames → снимайте медленнее, добавьте texture и overlap;
- не видны верх/низ → добавьте высотные пояса;
- заполнена полость → снимите вид, прямо наблюдающий её;
- неверный mask → исправьте target tracking;
- неверные размеры → добавьте известную длину или внешнюю metric calibration;
- много кадров с одной дуги → это всё ещё плохое camera coverage.

Подробности фото-пути:
[INTERNET_PHOTO_TO_CAD.md](INTERNET_PHOTO_TO_CAD.md).
