# 자연어 기반 인공지능 산업분류 자동화 공모전

### 목적 및 분석 조건
- 목적 : 자연어 기반의 통계데이터를 인공지능으로 자동 분류하는 기계학습 모델 발굴로 통계 데이터 활용 저변 확대
- 분석 조건 : Python 또는 R 언어로 자연어 텍스트 마이닝 및 인공지능 분류 모델 구축

### 분석 환경 및 성과
- 구축 모델 : KoBERT ([SKTBrain](https://github.com/SKTBrain/KoBERT))
- 구동환경 : google colab (pro 버젼)
- 약 500 여 팀 중 37위로 마감 (최종 accuracy : 89.8, f1-score(macro) : 76.44)

### 모델 구축 프로세스 
(포토샵 이용해서 구축 프로세스 시각화)
1. 텍스트 데이터 전처리 (py-hanspell) 
2. 불균형 문제 해결 (text-aumentation)
3. 학습 데이터 분할 후, 모델 학습 (Kobert)
4. 예측된 소분류 값을 이용하여 대/중분류 mapping 하여, 3가지 최종 예측값 도출  

```python
#구글드라이브 연동
from google.colab import drive
drive.mount('/content/drive')

# gpu 켜기
import torch
device = torch.device("cuda:0")

# 저장 경로 미리 지정
path = '/content/drive/MyDrive/nlp_c/'
```

```python
#깃허브에서 KoBERT 파일 로드
!pip install ipywidgets  # for vscode
!pip install git+https://git@github.com/SKTBrain/KoBERT.git@master

# 필요한 모듈 로딩
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm.notebook import tqdm
from tqdm import tqdm_notebook

#kobert
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

#transformers
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
```

```python
#BERT 모델, Vocabulary 미리 불러오기
bertmodel, vocab = get_pytorch_kobert_model()
```


