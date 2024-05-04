# ai-tetris-demo

딥러닝 강화학습을 이용한 AI 테트리스 프로젝트

by beyond ZERO

## Setup

```bash
rye sync
```

## Run

```bash
rye run train # cpu
rye run train-cuda # cuda (gpu)
rye run train-mps # metal (gpu)
```

## Simulate

```bash
# if you want to change the model, please edit `WEIGHT_OUT_PATH` in `src/tetris_project/ai/NN.py`
rye run simulate
```

## Test

```bash
rye run test
```
