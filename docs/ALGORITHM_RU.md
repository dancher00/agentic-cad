# Как работает DA3-CAD

## Коротко

Текущий основной путь v9:

```text
RGB-ракурсы одного неподвижного предмета + маски выбранного экземпляра
        |
exhaustive COLMAP SfM
        |
калиброванные камеры + одинаковое undistort RGB и масок
        |
masked PatchMatch MVS
        |
cross-view-confirmed raw fused cloud
        |                              \
измеренные CAD-корни                  Poisson render
revolve / sketch-extrusion             |
        |                              CADENA proposal
        +---------------+--------------+
                        |
             единый конкурс кандидатов
                        |
       проекция в исходные calibrated views
                        |
 silhouette + depth + appearance-edge gates
                        |
       OpenCascade: valid single-solid STEP
                        |
                 ACCEPT / ABSTAIN
```

DA3 больше не является источником камер или основной геометрии. В плотном
маршруте камеры восстанавливает COLMAP, глубину — calibrated PatchMatch. DA3
остаётся опциональным экспериментальным prior для sparse/textureless 2DGS.

## Входной контракт

Нужны фотографии одного физического экземпляра:

- предмет и фон неподвижны, движется камера;
- соседние виды перекрываются на 60–80%;
- сняты противоположные стороны, верхние и нижние косые ракурсы;
- для каждого кадра есть mask выбранного экземпляра;
- пользователь знает, какой экземпляр выбран, но не обязан знать его CAD-класс.

Практический минимум — около 30 резких видов. Большее число кадров полезно,
только если оно добавляет новые направления наблюдения.

Маска — это target selection, а не классификация. Её можно получить вручную,
трекером или SAM2 из box/prompt. В benchmark используются раскрытые oracle masks,
чтобы отдельно измерять геометрию и не смешивать её с ошибкой сегментации.

## Этапы

1. **SfM.** COLMAP с exhaustive matching связывает неупорядоченные фотографии,
   восстанавливает intrinsics/extrinsics и регистрирует виды. RGB и masks
   undistort-ятся одной и той же моделью камеры.

2. **Плотная измеренная геометрия.** Masked PatchMatch строит глубину только для
   объекта. Fusion сохраняет точки, подтверждённые несколькими видами.

3. **Два представления геометрии.**
   `fused_cloud.ply` — исходное измерение для CAD fitter. Poisson mesh —
   сглаженный conditioning render для proposer. Он может выглядеть менее плотным
   и не имеет права заменять raw cloud.

4. **Кандидаты CAD без словаря деталей.**
   Измерительный fitter строит произвольный осевой профиль для `revolve` и
   произвольный замкнутый 2D-профиль для `sketch-extrusion`. Restricted CADENA
   может предложить дополнительную программу. Это грамматика операций, а не
   классы «кружка», «кронштейн» или «вал».

5. **Честная топология.** Pooled cloud потерял per-view identity, поэтому по нему
   нельзя доказать внутреннюю стенку. Shell, cavity, handle и другие составные
   операции должны появляться только из view-preserving evidence. Невидимая
   геометрия не объявляется измеренной.

6. **Проверка исходными изображениями.** Каждый кандидат проецируется обратно во
   все пригодные calibrated views. Проверяются silhouette IoU, depth inliers и
   внутренние appearance edges. Плохой MVS-view не считается измерением, если в
   нём менее 128 depth pixels или depth покрывает менее 10% target mask.

7. **B-Rep.** CadQuery-программа выполняется в sandbox. OpenCascade проверяет
   `isValid()`, положительный объём и ровно один solid, затем экспортирует STEP.

## Что означает результат

- `model.step` появляется только после kernel и source-view acceptance;
- `candidate.step` сохраняется при `ABSTAIN` для аудита;
- валидный STEP ещё не означает правильную геометрию;
- линии и треугольники в STL-preview — тесселяция визуализатора, не CAD-грани.

Текущие frozen source-view пороги: silhouette IoU 0.87, depth inlier fraction
0.90, appearance-edge precision 0.45 и recall 0.12, когда внутренние границы
наблюдаемы.

## Проверенный real-RGB benchmark

Пять T-LESS объектов прогнаны по 32 RGB-вида каждый. Все камеры
зарегистрированы; пять STEP-кандидатов kernel-valid и состоят из одного solid.
Но все пять остаются `ABSTAIN`, потому что нарушают хотя бы один source-view
gate.

| Case | Выбранный CAD-корень | Direct IoU | Решение |
|---|---|---:|---:|
| o02-fixed | measured revolve | 0.310 | ABSTAIN |
| o04-fixed | measured revolve | 0.368 | ABSTAIN |
| o10 | measured sketch-extrusion | 0.162 | ABSTAIN |
| o20-fixed | measured sketch-extrusion | 0.398 | ABSTAIN |
| o25 | restricted CADENA | 0.644 | ABSTAIN |

Средний no-alignment IoU v8→v9 вырос с 0.3263 до 0.3763, а
CD²×1000 снизился с 21.99 до 17.31. Это инженерный аудит пяти объектов, не SOTA
и не доказательство универсальности.

Машиночитаемый ledger:
[real-photo-e2e-v1.json](results/real-photo-e2e-v1.json).
Локальный семистраничный PDF с RGB, raw cloud, candidate и reference строится:

```bash
python scripts/build_real_photo_e2e_report.py
```

T-LESS RGB и производные изображения остаются в ignored `outputs/` и не
распространяются.

## Что пока не решено

- мелкие отверстия, фаски, скругления, резьбы и элементы ниже stereo resolution;
- shell/handle и многотельные композиции без view-preserving fitter;
- прозрачные, зеркальные, тонкие и движущиеся объекты;
- исходный feature tree, design intent, допуски и GD&T;
- миллиметровый масштаб без известного размера или внешней калибровки;
- скрытая геометрия, которой нет ни на одном снимке.

Подробный запуск:
[INTERNET_PHOTO_TO_CAD.md](INTERNET_PHOTO_TO_CAD.md).
