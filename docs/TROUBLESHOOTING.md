# Troubleshooting

Ниже описан основной dense path:
`prepare-photos-sfm → prepare-target → dense-surface → fit-cad`.
Legacy-команда `reconstruct` с DA3 depth имеет отдельные исследовательские
артефакты и не является рекомендуемым фото→CAD маршрутом.

## Python и окружения

Используйте CPython 3.12 и checked-in constraints:

```bash
conda create --prefix ./.venv python=3.12 pip -y
conda activate "$PWD/.venv"
python -m pip install -r constraints/cpu-py312.txt
python -m pip install -r constraints/cu130-py312.txt
python -m pip install -r constraints/cadena-py312.txt
python -m pip install --no-deps -e .
```

PatchMatch живёт в отдельном CUDA 12 environment, чтобы не заменять CUDA 13
runtime основного Torch:

```bash
python -m pip install "virtualenv>=20,<21"
scripts/setup_mvs_env.sh .venv/bin/python
.venv-mvs/bin/pip check
```

RTX 5080 16 GB достаточна для текущего inference. H100 ускоряет sweep, но не
исправляет плохие камеры, маски или ненаблюдаемую топологию.

## COLMAP регистрирует мало кадров

Проверьте `camera_recovery.json`. Для фотографий используйте
`--pairing exhaustive`; `sequential` подходит только упорядоченному видео.

Снимайте медленнее, держите 60–80% overlap, фиксируйте zoom/focus и оставляйте
текстурный неподвижный фон. Добавьте противоположную сторону, верхний и нижний
пояса. Несколько десятков кадров с одной дуги не дают полного coverage.

Turntable нарушает текущий SfM-контракт: фон неподвижен, а предмет движется.
Нужна moving-camera съёмка или отдельный object-centric pose estimator.

## Нет исходных masks

`prepare-photos-sfm --masks` принимает source-resolution PNG и undistort-ит
их точно той же моделью, что RGB. Если masks ещё нет, опустите эту опцию,
восстановите камеры и создайте masks или SAM2 boxes для
`registered_frames/`. Затем передайте их в `prepare-target`.

Имена masks должны совпадать со stem соответствующих изображений. Всегда
просматривайте overlay: ошибка target selection превращается в ошибку
геометрии, а не исправляется CAD fitter.

## PatchMatch не запускается

Проверьте:

```bash
.venv-mvs/bin/python -c "import torch; print(torch.cuda.is_available())"
.venv-mvs/bin/pip check
```

`dense-surface` заранее отказывает при camera coverage ниже 0.25. Не ослабляйте
gate только ради запуска: добавьте отсутствующие направления съёмки.

Если отдельный source view содержит менее 128 измеренных depth pixels или depth
покрывает менее 10% target mask, он исключается из CAD verification и
записывается в отчёт.

## Raw cloud выглядит лучше Poisson surface

Это ожидаемо. `fused_cloud.ply` хранит cross-view-confirmed измеренные точки и
является входом CAD fitter. `surface.ply` — сглаженный Poisson conditioning
render для proposer; он может быть менее плотным, терять отверстия и быть
non-watertight. Не подменяйте им `--measurements fused_cloud.ply`.

## STEP валиден, но геометрия неверна

OpenCascade проверяет только B-Rep-инварианты: один solid, положительный объём,
`isValid()` и экспорт. Геометрически неверный STEP тоже может пройти kernel.

Поэтому `fit-cad` дополнительно проецирует кандидата в исходные calibrated
views и проверяет silhouette, depth и appearance edges. При провале получается
`candidate.step` + `ABSTAIN`, а не успешный `model.step`. Причины находятся
в `cadena_report.json`.

## Пропало отверстие, shell или ручка

Pooled point cloud не сохраняет per-view identity. Он может поддержать внешний
revolve/sketch профиль, но не доказывает внутреннюю стенку. Поэтому прямой
revolve root намеренно остаётся solid.

Для cavity, shell, handle и составных тел нужен view-preserving fitter:
наблюдение внутренней поверхности из нескольких calibrated views, согласованные
границы и отрицательное пространство. Добавьте near-axis и oblique views.
Невидимая полость не должна достраиваться как измеренная.

## В preview много треугольников и линий

STL — тесселяция B-Rep для показа. Треугольники не означают, что STEP состоит из
тысяч CAD-граней. Авторитетны `candidate.step`/`model.step` и
`kernel_validation` в отчёте.

Если kernel report сам показывает десятки лишних faces/edges, это уже ошибка
CAD-программы или boolean topology, а не визуализатора.

## Размеры не в миллиметрах

COLMAP/MVS восстанавливают сцену с неизвестным similarity scale. До известного
размера или внешней metric calibration результат остаётся в canonical units.
Увеличение числа фотографий само по себе миллиметровый масштаб не создаёт.

## Output directory уже существует

Команды reconstruction не смешивают новое evidence со старым и поэтому
отказываются писать в существующий output directory. Выберите новый путь или
осознанно перенесите прежний run; не объединяйте артефакты вручную.

## Когда нужен DA3

DA3 не нужен dense benchmark и не является источником камер. Его текущая
экспериментальная роль — confidence-aware prior внутри sparse/textureless 2DGS.
Он выключен по умолчанию, пока held-out multi-seed проверка не подтвердит
выигрыш. См. [BREPGAUSSIAN_DA3.md](BREPGAUSSIAN_DA3.md).

Для воспроизведения исторических DA3 ledgers нужны pinned source/checkpoints и
явное принятие их лицензий; это не требуется CPU smoke или текущему dense path.
