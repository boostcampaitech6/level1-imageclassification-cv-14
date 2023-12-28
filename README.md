# [P Stage1] 도담도담

## 📋 Table of content

- [Project Overview](#Overview)<br>
- [팀 목표](#Team)<br>
- [File Tree](#filetree)<br>
- [실행 방법](#Code)<br>
- [Objective](#Objective)<br>
- [DataSet](#DataSet)<br>
- [MEMBER](#Member)



<br></br>
## :pencil2:Project Overview <a name = 'Overview'></a>

- COVID-19 확산으로 인해 마스크의 중요성이 대두됨
- 바이러스는 주로 입과 호흡기로 전파되기 때문에 코와 입을 완전히 가리는 올바른 방법으로 마스크를 착용 해야함
- 모든 사람이 마스크를 착용하여 전차 경로를 차단
- 그러나 모든 사람의 마스크 착용 상태를 확인하는 것은 인력적 제약이 따름
  ### => 카메라를 통해 사람의 얼굴 이미지만으로 마스크 착용 여부를 자동으로 판별할 수 있는 시스템의 개발이 필요


<br></br>
## :checkered_flag:팀 목표 <a name = 'Team'></a>
- 깃헙을 사용하여 협업 능력 기르기
- 파이토치 템플릿을 활용하여 프로젝트 구조화 시키는 능력 기르기


<br></br>
## :deciduous_tree:File Tree <a name = 'filetree'></a>
```
${level1-imageclassification-cv-14}
  │
  ├── train.py - main script to start training
  │
  ├── ensemble.py
  │
  ├── inference.py
  │
  ├── config.json - holds configuration for training
  ├── parse_config.py - class to handle config file and cli options
  │
  ├── test_config.json
  │
  ├── base/ - abstract base classes
  │     ├── base_dataset.py
  │     ├── base_model.py
  │     └── base_trainer.py
  │
  ├── data_loader/ - anything about data loading goes here
  │     ├── data_loaders.py
  │     ├── test_dataset.py
  │     ├── train_dataset.py
  │     └── valid_dataset.py
  │
  ├── model/ - models, losses, and metrics
  │     ├── model.py
  │     ├── multi_tast_model.py
  │     ├── metric.py
  │     └── loss.py
  │
  ├── saved/
  │     ├── models/ - trained models are saved here
  │     └── log/ - default logdir for tensorboard and logging output
  │
  ├── trainer/ - trainers
  │     ├── trainer.py
  │     ├── multi_tast_trainer.py
  │     └── one_tast_trainer.py
  │
  ├── logger/ - module for tensorboard visualization and logging
  │     ├── logger.py
  │     └── logger_config.json
  │  
  └── utils/ - small utility functions
         ├── util.py
         ├── copy_files.py
         └── data_util.py
```

<br></br>
## 💻실행 방법 <a name = 'Code'></a>
- 패키지 설치
  ```
  pip install -r requirements.txt
  ```


- Model training
  ```
  python train.py -c train_config.json
  ```
  - kfold train
    
    train_config.json에 인자 추가
    ```
    "train_set": {
     ...
    },
    "use_kfold": true,
    "k_splits": 5
    },
    ```

- Inference
  - single_model
    ```
    python inference.py -c test_config.json
    ```
  - ensemble
    ```
    python ensemble.py -c test_config.json
    ```

    test_train.json에 인자 추가
    ```
    "multi_model": {
        ...
        ,   
        "is_multi_task": true, - output features가 8개인지 18개인지를 확인하는 변수
        "is_deep_model": true  - 각 task가 다른 모델로 학습하는 지를 확인하는 변수
    }
    ```
 - Visualization Result
   - wandb
  
     train_config.json
     ```
     "wandb": {
        "exp_name": "EXP_NAME",
        "exp_num": 0,
        "project_name": "Image_Classification",
        "entity": "cv-14"
     }
     ```


<br></br>
## :chess_pawn:Objective <a name = 'Objective'></a>
성별, 연령, 마스크 착용 여부에 따라 사진을 총 18개의 클래스로 분류


<br></br>
## :minidisc:DataSet <a name = 'DataSet'></a>

- 전체 사람 명 수 : 4,500
- 한 사람당 사진의 개수: 7 [마스크 착용 5장, 이상하게 착용(코스크, 턱스크) 1장, 미착용 1장]
- 이미지 크기: (384, 512)


### Class Description

|class|mask|gender|age|
|---|---|---|---|
|0|Wear|Male|<30|
|1|Wear|Male|>=30 and <60|
|2|Wear|Male|>=60|
|3|Wear|Female|<30|
|4|Wear|Female|>=30 and <60|
|5|Wear|Female|>=60|
|6|Incorrect|Male|<30|
|7|Incorrect|Male|>=30 and <60|
|8|Incorrect|Male|>=60|
|9|Incorrect|Female|<30|
|10|Incorrect|Female|>=30 and <60|
|11|Incorrect|Female|>=60|
|12|Not Wear|Male|<30|
|13|Not Wear|Male|>=30 and <60|
|14|Not Wear|Male|>=60|
|15|Not Wear|Female|<30|
|16|Not Wear|Female|>=30 and <60|
|17|Not Wear|Female|>=60|  




---
## :trophy:MEMBER <a name = 'Member'></a>

|김실희|김정택|김채아|선경은|
|:--:|:--:|:--:|:--:|
|:whale:|:chipmunk:|:hatched_chick:|:penguin:|
