# Nature DQN (2015), PyTorch

Mnih et al., **Human-level control through deep reinforcement learning**
(Nature, 2015)의 CNN과 학습 알고리즘 구현입니다. 기존 `model.py`의
`CNNDQN`을 재사용하며 `NatureDQN`이 입력 검증과 uint8 정규화를 추가합니다.

## 파일

- `nature.py`: `NatureDQN`, `DQNConfig`, `NatureRMSprop`, `ReplayMemory`, `DQNAgent`.
- `train_nature.py`: Atari/CartPole 이미지 환경 생성, 학습, 체크포인트 저장, 평가 CLI.
- `test_nature.py`: ROM 없이 실행하는 수치 및 학습 루프 회귀 테스트.
- `requirements.txt`: 이 구현에 필요한 패키지.

## 네트워크 사용

저장소 루트에서 실행합니다.

```python
import torch
from dqn.nature import NatureDQN

model = NatureDQN(n_actions=4)
state = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)
q_values = model(state)  # shape: (2, 4)
actions = q_values.argmax(dim=1)
```

입력은 `(batch, 4, 84, 84)`입니다. uint8 `[0, 255]`는 모델 내부에서
`255`로 나누며, float 입력은 이미 `[0, 1]`로 정규화되어 있어야 합니다.

| 층 | 출력 / 설정 |
| --- | --- |
| 입력 | 흑백 84×84 프레임 4개, 오래된 프레임부터 |
| Conv1 + ReLU | 32채널, kernel 8, stride 4 → 32×20×20 |
| Conv2 + ReLU | 64채널, kernel 4, stride 2 → 64×9×9 |
| Conv3 + ReLU | 64채널, kernel 3, stride 1 → 64×7×7 |
| Flatten → Linear + ReLU | 3136 → 512 |
| Linear | 512 → 행동 수, Q-value 출력 |

## 학습 알고리즘과 기본값

```text
y = clipped_reward + 0.99 * (1 - terminal) * max_a Q_target(next_state, a)
```

| 항목 | 설정 |
| --- | --- |
| Replay | 균등 무작위 표본, 최대 1,000,000 transitions |
| Minibatch | 32 |
| 학습 시작 | 50,000 agent steps 이후 |
| 학습 주기 | agent steps 4회마다 1회 업데이트 |
| Target network | agent steps 10,000회마다 hard copy |
| Learning rate | 0.00025 |
| 탐험 | warmup 중 ε=1, 이후 1,000,000 steps 동안 1→0.1 선형 감소 |
| 평가 탐험 | ε=0.05 |
| Reward clipping | 학습 보상만 [-1, 1], 출력 점수는 원래 보상 합 |
| TD-error clipping | Huber(delta=1), minibatch 합으로 역전파 |
| RMSProp | centered, decay=0.95, epsilon=0.01을 제곱근 **안**에 적용 |
| 학습 길이 | 50,000,000 agent steps |

스케줄 단위는 **행동 결정 횟수(agent steps)**입니다. 한 결정은 최대
4개의 ALE 프레임에 대응합니다. Target 주기는 optimizer 업데이트 횟수가
아니며, 원본 공개 실행 설정과 learner의 step 카운터를 기준으로 했습니다.

Atari는 환경 자체의 frameskip을 1로 두고 `AtariPreprocessing`에서
action repeat=4, 마지막 두 프레임 max pooling, 흑백 84×84 변환,
reset 시 1~30 no-op을 적용합니다. Sticky actions는 0, 행동 집합은 minimal입니다.
프레임 history의 빈 자리는 0으로 채웁니다.

학습 중 life loss는 TD target의 terminal로 처리하고 history를 초기화하지만,
게임 전체를 reset하지 않습니다. 실제 종료나 time limit에는 환경을 reset합니다.
Time limit(`truncated`)은 종료 직전 상태에서 bootstrap을 유지합니다.
평가는 life loss에 멈추지 않고 전체 게임 점수를 기록합니다.

## 설치와 실행

```bash
# 저장소 루트, 프로젝트 가상환경
.venv/bin/python -m pip install -r dqn/requirements.txt

# 학습: CUDA → MPS → CPU 순으로 사용 가능한 장치 자동 선택
.venv/bin/python -m dqn.train_nature --env ALE/Breakout-v5

# 짧은 동작 확인: 학습 성능 평가용 설정이 아님
.venv/bin/python -m dqn.train_nature \
  --env ALE/Breakout-v5 --device cpu \
  --total-steps 256 --learning-starts 32 --replay-capacity 256 \
  --log-interval 64 --checkpoint /tmp/nature-dqn-smoke.pt

# 저장한 모델 평가 (--env는 체크포인트의 환경과 같아야 함)
.venv/bin/python -m dqn.train_nature \
  --env ALE/Breakout-v5 --evaluate --episodes 10 \
  --checkpoint dqn/checkpoints/nature.pt

# 화면 표시 평가
.venv/bin/python -m dqn.train_nature --evaluate --render --episodes 1

# ROM이 필요 없는 테스트
.venv/bin/python -m pytest dqn/test_nature.py -q
```

학습 중 125,000 steps마다, 그리고 정상 종료 시 체크포인트를 저장합니다.
온라인/target 가중치, optimizer 상태, 설정, step, 환경 ID가 포함됩니다.
`--checkpoint`는 학습 시 저장 경로이고 `--evaluate`에서는 읽을 경로입니다.
Replay, RNG, 에뮬레이터 상태를 저장하지 않으므로 정확한 중단 재개 기능은
제공하지 않습니다. 평가 없이 실행하면 새 모델로 학습을 시작합니다.

Replay는 CPU에 uint8 프레임을 한 번씩 저장하고 인접 transition끼리 공유합니다.
기본 100만 transition에는 프레임 본체만 약 6.6 GiB가 필요하고 Python 객체,
episode 경계 및 minibatch 메모리가 추가됩니다. 작은 메모리 환경에서는
`--replay-capacity 100000` 등으로 줄일 수 있지만 원본 실험 설정과 달라집니다.

## TensorBoard

학습 CLI는 기본적으로 `runs/nature_dqn/` 아래에 환경 이름, 실행 시각, seed로
구분된 로그 폴더를 생성합니다. `--log-dir`로 상위 폴더를 지정할 수 있습니다.
평가 모드(`--evaluate`)는 학습 로그를 생성하지 않습니다.

| Scalar 이름 | 내용 | 가로축 |
| --- | --- | --- |
| `train/loss` | 해당 minibatch의 평균 Huber loss, 매 학습 업데이트 기록 | Agent step |
| `train/reward` | clipping 전 보상의 에피소드 합, 종료·시간 제한 시 기록 | Episode 번호 |

태그와 가로축 종류는 `dqn/cart_pole.py`의 TensorBoard 형식에 맞췄습니다.
기존 `runs/cnndqn-hardcopy`와 새 실행을 같은 그래프에서 비교하려면
`.venv/bin/tensorboard --logdir runs --port 6006`으로 실행하세요.
태그 변경 전에 생성한 이벤트 파일은 기존 태그를 유지합니다.

Loss는 warmup이 끝나고 실제 업데이트가 시작된 뒤 나타납니다. Loss 그래프는
평균값을 표시하지만, 원본에 맞춘 역전파의 minibatch 합 연산은 유지합니다.
Reward는 전체 에피소드가 끝나야 기록되며 Atari life loss만으로는 기록하지
않습니다. `--log-interval`은 콘솔 출력 주기이며 TensorBoard 기록 주기와는
별개입니다. 로그는 약 10초 간격으로 flush하고 종료·예외·Ctrl+C 시 writer를
닫아 남은 기록을 저장합니다.

```bash
# 다른 터미널에서 실행하고 http://localhost:6006 에 접속
.venv/bin/tensorboard --logdir runs/nature_dqn --port 6006
```

TensorBoard의 **Scalars**에서 두 지표를 선택하세요. 학습을 여러 번 실행하면
각 run의 곡선을 비교할 수 있습니다. 예를 들어 CartPole 로그를 별도로
모으려면 학습 명령에 `--log-dir runs/cartpole`을 붙이고 TensorBoard도
`--logdir runs/cartpole`로 실행하면 됩니다. 기존 실행의 기록은 소급 생성되지
않으며, 이 변경 이후 시작하는 학습부터 기록합니다.

Python에서 `train()`을 직접 호출할 때는 `SummaryWriter`를 `writer=`로
전달하고 호출 측에서 닫아주세요. CLI에서는 생성과 종료를 자동 처리합니다.

## CartPole 실행

`--env CartPole-v1`을 지정하면 CartPole의 렌더링 화면을 84×84 흑백 이미지로
변환합니다. Replay가 프레임 4장을 쌓아 동일한 Nature DQN CNN에 입력합니다.
기본 숫자 관측값 4개를 사용하는 MLP 방식은 아닙니다.

전처리 함수는 `dqn/cart_pole.py`의 Gymnasium wrapper를 사용하고,
처리 순서와 history 초기화 규칙은 Nature DQN 기준으로 유지합니다.

```text
CartPole RGB 화면
→ AddRenderObservation(render_only=True)
→ GrayscaleObservation(keep_dim=False)
→ ResizeObservation((84, 84))
→ Replay에서 최근 4프레임 구성 (초기 상태: [0, 0, 0, 첫 프레임])
→ NatureDQN에서 float32 / 255.0
```

`dqn/cart_pole.py`의 resize→grayscale 순서와 `FrameStackObservation`의
기본 reset-frame 반복을 그대로 복사하지는 않습니다. 프레임 쌓기와 정규화는
중복 적용하지 않으며, 학습 replay와 평가 history에 같은 zero padding을
적용합니다. 테스트에서는 위 순서의 Gymnasium wrapper에
`FrameStackObservation(padding_type="zero")`, `TransformObservation`을
연결한 결과와 모델 입력값이 reset/step마다 일치하는지 확인합니다.

이것은 Lua 이미지 연산의 픽셀 단위 재현은 아닙니다. 원본 `Scale.lua`는
`image.rgb2y`와 bilinear resize를 사용하지만 Gymnasium의 grayscale 계수와
`ResizeObservation`의 OpenCV `INTER_AREA` 보간은 다릅니다.

CartPole은 행동당 physics step 1회이며 Atari의 action repeat, max pooling,
random no-op, life-loss 처리를 적용하지 않습니다. `CartPole-v1`의 기본
500-step time limit을 유지하고 시간 제한 시 bootstrap을 유지합니다.

```bash
# 의존성: classic-control extra가 화면 생성용 pygame을 설치
.venv/bin/python -m pip install -r dqn/requirements.txt

# CartPole 이미지 기반 학습 예시 (Nature 기본값보다 짧은 실험 설정)
.venv/bin/python -m dqn.train_nature \
  --env CartPole-v1 --total-steps 100000 \
  --learning-starts 1000 --replay-capacity 10000 \
  --exploration-steps 50000 --target-update-interval 1000 \
  --log-interval 1000 --checkpoint dqn/checkpoints/cartpole.pt

# 학습한 모델 평가, 화면 표시
.venv/bin/python -m dqn.train_nature \
  --env CartPole-v1 --evaluate --episodes 5 --render \
  --checkpoint dqn/checkpoints/cartpole.pt

# 짧은 실행 확인
.venv/bin/python -m dqn.train_nature \
  --env CartPole-v1 --total-steps 256 --learning-starts 32 \
  --replay-capacity 256 --exploration-steps 128 \
  --target-update-interval 64 --device cpu --log-interval 64 \
  --checkpoint /tmp/nature-cartpole-smoke.pt
```

`--render` 없이도 CNN 입력을 위해 내부적으로 RGB 화면을 생성합니다.
디스플레이가 없는 서버에서는 명령 앞에 `SDL_VIDEODRIVER=dummy`를 붙입니다.
실제 창을 보려면 이 환경 변수를 해제하고 `--render`를 사용합니다.
위 학습 예시는 실행을 위한 시작 설정이며 CartPole 해결 성능을 검증한
하이퍼파라미터는 아닙니다. 환경별로 체크포인트 경로를 구분해 사용하세요.

## 재현 범위와 참고 자료

이 코드는 학습 가능한 PyTorch 포트이며 논문 점수 재현을 보장하지 않습니다.
현대 Gymnasium/ALE의 전처리, resize, ROM, 초기화와 게임 종료 동작은
원래 Lua/Torch 환경과 비트 단위로 같지 않습니다. 평가에는 108,000 raw-frame
time limit을 사용하며, 원본 전체 49게임 벤치마크와 human-start 평가는 포함하지
않았습니다. PyTorch의 기본 파라미터 초기화를 사용합니다.

- [논문, Nature 518, 529–533](https://www.nature.com/articles/nature14236)
- [DeepMind 원본 저장소](https://github.com/google-deepmind/dqn)
- [원본 CNN 설정](https://github.com/google-deepmind/dqn/blob/master/dqn/convnet_atari3.lua)
- [원본 learner: TD target, RMSProp, 탐험 스케줄](https://github.com/google-deepmind/dqn/blob/master/dqn/NeuralQLearner.lua)
- [원본 실행 하이퍼파라미터](https://github.com/google-deepmind/dqn/blob/master/run_cpu)
- [원본 흑백 변환 및 resize 순서](https://github.com/google-deepmind/dqn/blob/master/dqn/Scale.lua)
- [원본 history zero padding과 정규화](https://github.com/google-deepmind/dqn/blob/master/dqn/TransitionTable.lua)
- [Gymnasium Atari 전처리](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.AtariPreprocessing)
