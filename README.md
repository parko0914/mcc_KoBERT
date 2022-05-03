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

### 1. text cleaning
([pyhanspell 원문](https://github.com/ssut/py-hanspell))
* 네이버 맞춤법 교정기 기반 라이브러리인 pyhanspell 을 이용
* 모든 데이터를 교정기를 통해 맞춤법을 교정
* 형태소 처리를 하지 않은 이유 : BERT 모델의 경우 문장의 앞뒤 문맥까지 파악하여 고려해주기 때문에 형태소로 분리하지 않고, 맞춤법 교정을 통해 더 정확한 데이터로 train 하기 위함

```python
# 맞춤법 교정 함수
def comment_clean_t(data):
    comment = data['clean']
    comment_list = []
    for i in tqdm(range(len(comment))):    
        try:
            # 특정 특수문자 삭제
            sent = comment[i]

            hanspell_sent = spell_checker.check(sent).checked
            comment_list.append(hanspell_sent)
        except:
            comment_list.append(sent)
    return comment_list
```

### 2. text data augmentation 
([EDA 참고](https://catsirup.github.io/ai/2020/04/21/nlp_data_argumentation.html))
* target 변수의 각 class 빈도수가 불균형한 경우, 모델의 성능이 현저히 낮아질 가능성이 높음
* n 수가 500개 이하인 class 에 해당하는 데이터에 대해 EDA(Data Augumentation) 진행

```python
# text augmentation
# pip install -U nltk
import nltk; 
nltk.download('omw-1.4');
# nltk.download('wordnet') # 영문 버전
# eda 폴더 생성
% cd /content/drive/MyDrive/nlp_c
# !git clone https://github.com/jasonwei20/eda_nlp
# !git clone https://github.com/catSirup/KorEDA
# eda는 eda_nlp/code 폴더에, wordnet.pickle 은 eda_nlp 폴더로 이동시키고, 진행
% cd eda_nlp/
# 추가적으로 augment.py 64번째 항에 eda -> EDA로 변경해야 실행됨
s_class_n = pd.DataFrame(train_d['y_s'].value_counts().sort_values())
# s_class_n.to_csv(path + 'testset_class.csv', index=False, encoding='EUC-KR')
s_class = s_class_n[s_class_n['y_s'] < 500].index.tolist()
len(s_class)
# n수가 부족한 class aug_num 차등으로 증강(적은 순서대로 20, 10 ,5) -> 상대적으로 부족한 클래스 데이터 비율이 더 높아지는 것을 조금이라도 방지하고자 함
s_class1 = s_class[:30]
s_class2 = s_class[30:60]
s_class3 = s_class[60:]
few_d1 = train_d[train_d['y_s'].isin(s_class1)]
few_d2 = train_d[train_d['y_s'].isin(s_class2)]
few_d3 = train_d[train_d['y_s'].isin(s_class3)]
# n이 100개 이하인 클래스 뽑아서 augmentation 가능한 파일 형태로 만들어주기
txt_aug_list = [str(a) + '\t' + str(b) for a, b in zip(few_d1['label_s'], few_d1['clean_done'])]
with open(path + 'final_data/' + 'text_aug_1.txt', 'w') as f:
  f.write('\n'.join(txt_aug_list) + '\n')

txt_aug_list = [str(a) + '\t' + str(b) for a, b in zip(few_d2['label_s'], few_d2['clean_done'])]
with open(path + 'final_data/' + 'text_aug_2.txt', 'w') as f:
  f.write('\n'.join(txt_aug_list) + '\n')

txt_aug_list = [str(a) + '\t' + str(b) for a, b in zip(few_d3['label_s'], few_d3['clean_done'])]
with open(path + 'final_data/' + 'text_aug_3.txt', 'w') as f:
  f.write('\n'.join(txt_aug_list) + '\n')

# input file 형식 -> txt 파일 내 한 행 당 label + \t + text 형태로 들어간 파일 
# SR: Synonym Replacement, 특정 단어를 유의어로 교체
# RI: Random Insertion, 임의의 단어를 삽입
# RS: Random Swap, 문장 내 임의의 두 단어의 위치를 바꿈
# RD: Random Deletion: 임의의 단어를 삭제
!python code/augment.py --input=/content/drive/MyDrive/nlp_c/final_data/text_aug_1.txt --output=/content/drive/MyDrive/nlp_c/final_data/test_aug_eda_1.txt --num_aug=20 --alpha_sr=0.1 --alpha_rd=0.2 --alpha_ri=0.1 --alpha_rs=0.0
!python code/augment.py --input=/content/drive/MyDrive/nlp_c/final_data/text_aug_2.txt --output=/content/drive/MyDrive/nlp_c/final_data/test_aug_eda_2.txt --num_aug=10 --alpha_sr=0.1 --alpha_rd=0.2 --alpha_ri=0.1 --alpha_rs=0.0
!python code/augment.py --input=/content/drive/MyDrive/nlp_c/final_data/text_aug_3.txt --output=/content/drive/MyDrive/nlp_c/final_data/test_aug_eda_3.txt --num_aug=5 --alpha_sr=0.1 --alpha_rd=0.2 --alpha_ri=0.1 --alpha_rs=0.0
# augmentation 완료한 데이터 불러와서 기존데이터셋에 붙여주기(augmentation 대상 데이터는 삭제)
with open('/content/drive/MyDrive/nlp_c/final_data/test_aug_eda_1.txt', "r") as file:
  strings = file.readlines()
aug_d1 = pd.DataFrame([x.split('\n')[0].split('\t') for x in strings])
aug_d1.columns = ['label_s', 'clean_done']

with open('/content/drive/MyDrive/nlp_c/final_data/test_aug_eda_2.txt', "r") as file:
  strings = file.readlines()
aug_d2 = pd.DataFrame([x.split('\n')[0].split('\t') for x in strings])
aug_d2.columns = ['label_s', 'clean_done']

with open('/content/drive/MyDrive/nlp_c/final_data/test_aug_eda_3.txt', "r") as file:
  strings = file.readlines()
aug_d3 = pd.DataFrame([x.split('\n')[0].split('\t') for x in strings])
aug_d3.columns = ['label_s', 'clean_done']

train_d = train_d[train_d['y_s'].isin(s_class)==False]
sample_d = pd.concat([train_d[['label_s', 'clean_done']], aug_d1, aug_d2, aug_d3], axis = 0, ignore_index = True)
```
