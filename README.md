# 자연어 기반 인공지능 산업분류 자동화 공모전

### 목적 및 분석 조건
- 목적 : 자연어 기반의 통계데이터를 인공지능으로 자동 분류하는 기계학습 모델 발굴로 통계 데이터 활용 저변 확대
- 분석 조건 : Python 또는 R 언어로 자연어 텍스트 마이닝 및 인공지능 분류 모델 구축

### 분석 환경 및 성과
- 구축 모델 : KoBERT ([SKTBrain](https://github.com/SKTBrain/KoBERT))
- 구동환경 : google colab (pro 버젼)
- 약 500 여 팀 중 37위로 마감 (최종 accuracy : 89.99, f1-score(macro) : 74.58)

<img width="179" alt="KakaoTalk_20220427_160646204" src="https://user-images.githubusercontent.com/91238910/167256729-3c95c8a5-764a-4e22-b1c4-fc46c299b1dc.png">


### 모델 구축 프로세스 
![git_kobert](https://user-images.githubusercontent.com/91238910/182099435-6749f272-fa51-4e43-98a2-342c480e14ae.png)
1. 텍스트 데이터 전처리 (py-hanspell) 
2. 불균형 문제 해결 (text-aumentation)
3. 학습 데이터 분할 후, 모델 학습 (Kobert)
4. 예측된 소분류 값을 이용하여 대/중분류 mapping 하여, 3가지 최종 예측값 도출  

### 1. text cleaning
([pyhanspell 원문](https://github.com/ssut/py-hanspell))
* 네이버 맞춤법 교정기 기반 라이브러리인 pyhanspell 을 이용
* 모든 데이터를 교정기를 통해 맞춤법을 교정
* 형태소 처리를 하지 않은 이유 : BERT 모델의 경우 문장의 앞뒤 문맥까지 파악하여 고려해주기 때문에 형태소로 분리하지 않고, 맞춤법 교정을 통해 더 정확한 데이터로 train 하기 위함

#### 환경 구축
```python
!pip install git+https://github.com/ssut/py-hanspell.git
from hanspell import spell_checker
from tqdm import tqdm
import pandas as pd
```

#### 맞춤법 교정 함수
```python
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
* n 수가 500개 이하인 class 에 해당하는 데이터에 대해 EDA(Easy Data Augumentation) 진행

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
```

```python
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
```

```python
# input file 형식 -> txt 파일 내 한 행 당 label + \t + text 형태로 들어간 파일 
# SR: Synonym Replacement, 특정 단어를 유의어로 교체
# RI: Random Insertion, 임의의 단어를 삽입
# RS: Random Swap, 문장 내 임의의 두 단어의 위치를 바꿈
# RD: Random Deletion: 임의의 단어를 삭제
!python code/augment.py --input=/content/drive/MyDrive/nlp_c/final_data/text_aug_1.txt --output=/content/drive/MyDrive/nlp_c/final_data/test_aug_eda_1.txt --num_aug=20 --alpha_sr=0.1 --alpha_rd=0.2 --alpha_ri=0.1 --alpha_rs=0.0
!python code/augment.py --input=/content/drive/MyDrive/nlp_c/final_data/text_aug_2.txt --output=/content/drive/MyDrive/nlp_c/final_data/test_aug_eda_2.txt --num_aug=10 --alpha_sr=0.1 --alpha_rd=0.2 --alpha_ri=0.1 --alpha_rs=0.0
!python code/augment.py --input=/content/drive/MyDrive/nlp_c/final_data/text_aug_3.txt --output=/content/drive/MyDrive/nlp_c/final_data/test_aug_eda_3.txt --num_aug=5 --alpha_sr=0.1 --alpha_rd=0.2 --alpha_ri=0.1 --alpha_rs=0.0
```

```python
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

### 3. create Multi-class classification model(KoBERT)
* 전처리한 텍스트 데이터를 KoBERT 모델에 적용

#### train & test set 나누기
```python
from sklearn.model_selection import train_test_split
# stratify 를 target으로 지정해 비율을 맞춤으로써, 성능향상 가능
dataset_train, dataset_test, y_train, y_test = train_test_split(sample_d['clean_done'],
                               sample_d['label_s'], random_state=132, stratify=sample_d['label_s']) 

dataset_train = [[str(a), str(b)] for a, b in zip(dataset_train, y_train)]
dataset_test = [[str(a), str(b)] for a, b in zip(dataset_test, y_test)]
```

#### KoBERT model 환경, 데이터 맞춰주기
```python
# KoBERT 입력 데이터로 만들기
# BERT 모델에 들어가기 위한 dataset을 만들어주는 클래스
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

#BERT 모델, Vocabulary 불러오기
bertmodel, vocab = get_pytorch_kobert_model()

# Setting hyperparameters
max_len = 64
batch_size = 128
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate = 5e-5 # 0.0001 # 

# 토큰화 실행
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, pad = True, pair = False)
data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, pad = True, pair = False)

# torch 형식의 dataset 생성
# num_worker 은 gpu 활정화 정도, 5로 하니 오히려 과부화가 걸려 4로 조정
train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=4)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=4)
```

#### KoBERT model 환경, 데이터 맞춰주기
```python
# 클래스 수 조정
print(len(y_dict))

# KoBERT 학습모델 만들기
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=len(y_dict),   ##클래스 수 조정해줘야함##
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)
        
# model 선언 -> dr_rate : learning rate
model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)

# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

# optimizer 설정
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
# loss function 설정
loss_fn = nn.CrossEntropyLoss() # multi-class 이므로, binary X
# train total 수
t_total = len(train_dataloader) * num_epochs
# warmup_step
warmup_step = int(t_total * warmup_ratio)
# scheduler 
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

# 정확도 도출 함수
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc
```


#### 최종 모델 학습
```python 
highest_acc = 0
patience = 0

# 최종 모델 학습시키기
for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
    
    model.eval()

    for test_batch_id, (test_token_ids, test_valid_length, test_segment_ids, test_label) in enumerate(tqdm_notebook(test_dataloader)):
        test_token_ids = test_token_ids.long().to(device)
        test_segment_ids = test_segment_ids.long().to(device)
        test_valid_length= test_valid_length
        test_label = test_label.long().to(device)
        test_out = model(token_ids, valid_length, segment_ids)
        test_loss = loss_fn(out, label)
        test_acc += calc_accuracy(out, label)
    print("epoch {} test acc {}".format(e+1, test_acc / (test_batch_id+1)))
    
    # torch 형식 모델 저장

    if test_acc > highest_acc:
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': test_loss,
            }, path + 'final_data/' + 'correct_model_fin.pt')
        patience = 0
    else:
        print("test acc did not improved. best:{} current:{}".format(highest_acc, test_acc))
        patience += 1
        if patience > 5:
            break
    print('current patience: {}'.format(patience))
    print("************************************************************************************")
```

#### 예측값 도출
```python
# test 하기전 모델 기본 값 불러오기

# KoBERT 입력 데이터로 만들기
# BERT 모델에 들어가기 위한 dataset을 만들어주는 클래스
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

# 토큰화 실행
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

# KoBERT 학습모델 만들기
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=225,   ##클래스 수 조정해줘야함##
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

# Setting parameters
# 이건 나중에 최적화 값 찾아봐야할 듯
max_len = 64
batch_size = 128
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate = 5e-5 # 0.0001 # 


model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

checkpoint = torch.load(path + 'final_data/' + 'correct_model_fin.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

# 불러온 모델 평가 모드 실행
model.eval()
```

```python
from tqdm.notebook import tqdm

# 예측 함수 생성
def predict_set(dataset_test):

    test_acc = 0.0

    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)

    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=4)

    out_list =[]

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length= valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)
        output = out.detach().cpu().tolist()
        out_list.append(output)

    pd = sum(out_list,[])
    pd_list = pd_list = [np.argmax(i) for i in pd]
    return pd_list
```

```python
dataset_test = [[str(a), '0'] for a in test['clean_done']]
p_test = predict_set(dataset_test)
```

