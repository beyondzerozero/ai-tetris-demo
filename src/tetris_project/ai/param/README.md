# parametor 목록

## NN0

- Hold 없음
- 처음 성공시 parametor。
- $Dense(128) \rightarrow Dense(64) \rightarrow Dense(output\_size)$
- 손실함수 : Huber Loss
- $\epsilon\_{start} = 1.0, \ discount = 0.90, \ \epsilon\_{min} = 0.00001, \ \epsilon_{decay} = 0.995$
- Experience Buffer를 사용하며, 에이전트의 과거 행동결과를 축정해 두고, 그것을 이용하여 학습하는 것으로 NN학습을 정책오프로 실시한다. 경험의 재이용에 의한 데이터의 효율적인 이용 및 학습의 안정성 향상(수렴 시간 단축), 시계열 데이터에 있어서의 노이즈 배제를 기대할 수 있다.
- input은 아래

```python
def observe(self) -> np.ndarray:
    return np.concatenate([
        [
            self.line_total_count,
            self.get_hole_count(),
            self.get_latest_clear_mino_heght(),
            self.get_row_transitions(),
            self.get_column_transitions(),
            self.get_bumpiness(),
            self.get_eroded_piece_cells(),
            self.get_cumulative_wells(),
            self.get_aggregate_height(),
        ],
        self.current_mino_state.mino.to_tensor().flatten(),
        np.concatenate(
            [mino.to_tensor().flatten() for mino in self.mino_permutation][:NEXT_MINO_NUM]
        ),
    ])
```

## NN1

- Hold 기능 추가
- mino의 움직임 버그를 수정시 parametor이다. (이동후보를 전부 열거할 수 없는 버그)
- Model, parametor는 NN0과 동일함
- input은 아래

```python
def observe(self) -> np.ndarray:
    return np.concatenate([
        [
            self.line_total_count,
            self.get_hole_count(),
            self.get_latest_clear_mino_heght(),
            self.get_row_transitions(),
            self.get_column_transitions(),
            self.get_bumpiness(),
            self.get_eroded_piece_cells(),
            self.get_cumulative_wells(),
            self.get_aggregate_height(),
        ],
        self.current_mino_state.mino.to_tensor().flatten(),
        np.concatenate(
            [mino.to_tensor().flatten() for mino in self.mino_permutation][:NEXT_MINO_NUM]
        ),
        self.hold_mino.mino.to_tensor().flatten(),
    ])
```

## NN2

- mino の動きのバグを修正した時の parametor。( 移動候補を全列挙出来ていないバグ )
- $Dense(64) \rightarrow Dense(64) \rightarrow Dense(32) \rightarrow Dense(output\_size)$
- 損失関数 : Mean Squared Error
- $\epsilon\_{start} = 1.0, \ discount = 1.00, \ \epsilon\_{min} = 0.001, \ \epsilon_{decay} = 0.995$
- parametor の数は NN1 から減っているが、層を深くすることでより非線形な表現を可能にした。
- input NN1 と同じ

## NN3

- 盤面特徴量のバグ修正 & 新特徴量の追加
- Model, parametor は NN2 と同じ
- input は以下

```python
def observe(self) -> np.ndarray:
    return np.concatenate([
        [
            self.get_hole_count(),
            self.get_above_block_squared_sum(),
            self.get_latest_clear_mino_heght(),
            self.get_row_transitions(),
            self.get_column_transitions(),
            self.get_bumpiness(),
            self.get_eroded_piece_cells(),
            self.get_cumulative_wells(),
            self.get_aggregate_height(),
        ],
        self.current_mino_state.mino.to_tensor().flatten(),
        np.concatenate(
            [mino.to_tensor().flatten() for mino in self.mino_permutation][:NEXT_MINO_NUM]
        ),
        self.hold_mino.mino.to_tensor().flatten(),
    ])
```

## NN4

- NN3이라면 보드상단만으로 완결하려고 하는 $\rightarrow$ Experience Buffer를 상단, 하단에서 2개 준비해서 편향을 줄인다.
- Model, parametor, input은 NN2와 동일함

## NN5

- NN4의 parametor에서 스코어 안정성을 확보하기 위해 $\epsilon = 0.05$에 1 episode Max 3000점으로 Fine-tuning한다.
- Model, parametor, input은 NN2와 동일함

## NN6

- Pytorch로 마이그레이션함
- Model을 Batch Fit에서 1 Step Fit으로 변경한다. (torch 특성상)
