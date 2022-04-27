# 자연어 기반 인공지능 산업분류 자동화 공모전
- 목적 : 자연어 기반의 통계데이터를 인공지능으로 자동 분류하는 기계학습 모델 발굴로 통꼐 데이터 활용 저변 확대
- 분석 조건 : Python 또는 R 언어로 자연어 텍스트 마이닝 및 인공지능 분류 모델 구축

### KoBert 모델을 기반으로 한 모델 개발 
- 약 500 여 팀 중 37위로 마감
- 구동환경 : google colab (pro 버젼)

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


