### Introduction

This project is based on [libtorch-faster-rcnn](https://github.com/thisisi3/libtorch-faster-rcnn), and the motivation is also the same as what's discussed in there. This time, we hope to give one way of how one-stage object detector can be implemented using libtorch. 

### ATSS

The one-stage object detector we choose here is [ATSS](https://arxiv.org/abs/1912.02424). ATSS defines a way to assign positive and negative samples during training, and it can be applied to both anchor-based and anchor-free detectors. Here we implement the algorithm based on anchor-based one-stage detector---RetinaNet. We also add universal tricks like GN, centerness and GIoULoss. 

The overall setting and implementation very much follow mmdet's except for the positive-negative sample assignment part, in which this implementation is simpler and easier to follow. However, they produce the same results. 

### Compile and Use

#### compile with cmake(v3.19.2)

```shell
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=path_to_libtorch -DCMAKE_PREFIX_PATH=path_to_opencv
cmake --build . --config Release --parallel 8
```

#### train

```shell
./build/train configs/atss_r50_fpn_1x_voc.json --work-dir work_dir --gpu 0
```

#### inference

```shell
./build/test atss_r50_fpn_1x_voc.json epoch_12.pt --out epoch_12.bbox.json --gpu 0
```

**For more guidance please look at [readme](https://github.com/thisisi3/libtorch-faster-rcnn#readme).**

### Benchmark

Train: voc2007trainval + voc2012trainval

Test: voc2007test

Results are taking average of three runs.

|       | backbone | mAP  | AP50 |
| ----- | -------- | ---- | ---- |
| mmdet | Resnet50 | 53.3 | 78.1 |
| this  | Resnet50 | 53.4 | 78.1 |

