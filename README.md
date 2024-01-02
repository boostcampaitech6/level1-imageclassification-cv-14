# [P Stage1] 도담도담

## 📋 Table of content

- [대회 개요](#Overview)<br>
- [팀 소개](#Member)<br>
- [File Tree](#filetree)<br>
- [실행 방법](#Code)





<br></br>
## :pencil2:대회 개요 <a name = 'Overview'></a>

COVID-19 확산으로 인해 마스크의 중요성이 대두되었다. 모든 사람이 마스크를 잘 착용하였는지 마스크 착용 상태를 확인하는 것은 인력적 제약이 따르기 때문에, 카메라를 통해 사람의 얼굴 이미지만으로 마스크 착용 여부를 자동으로 판별할 수 있는 시스템의 개발이 필요하다.

  
 
  - Objective
    
    - 성별, 연령, 마스크 착용 여부에 따라 사진을 총 18개의 클래스로 분류

<br></br>
## :trophy:팀 소개 <a name = 'Member'></a>

|김실희|김정택|김채아|선경은|
|:--:|:--:|:--:|:--:|
|<a href='https://github.com/siL-rob'><img src='https://avatars.githubusercontent.com/u/58744783?v=4' width='200px'/></a>|<a href='https://github.com/Jungtaxi'><img src='https://avatars.githubusercontent.com/u/18082001?v=4' width='200px'/></a>|<a href='https://github.com/2018007956'><img src='https://avatars.githubusercontent.com/u/48304130?v=4' width='200px'/></a>|<a href='https://github.com/rudeuns'><img src='https://avatars.githubusercontent.com/u/151593264?v=4' width='200px'/></a>|


<br></br>
## :deciduous_tree:File Tree <a name = 'filetree'></a>
```
${level1-imageclassification-cv-14}
|
|-- train.py - main script to start training
|
|-- ensemble.py - script to run multiple models for making predictions
|
|-- inference.py - script to run the model for making predictions
|
|-- parse_config.py - class to handle config file and cli options
|
|-- test_config.json - holds the configuration for inference
|
|-- train_config.json - holds configuration for training
|
|-- requirements.txt - file listing the dependencies for the project
|
|-- README.md
|
|-- base - abstract base classes
|   |-- __init __.py
|   |-- base_dataset.py
|   |-- base_model.py
|   └── base_trainer.py
|
|-- data_aug_utils - configurations for data augmentation
|   |-- README.md
|   |-- aug_age.py
|   |-- aug_config.json
|   |-- aug_mask.py
|   └── rm_back.py
|
|-- data_loader - anything about data loading goes here
|   |-- __init __.py
|   |-- data_loader.py
|   |-- test_dataset.py
|   |-- train_dataset.py
|   └── valid_dataset.py
|
|-- model - models, losses, and metrics
|   |-- __init __.py
|   |-- loss.py
|   |-- metric.py
|   |-- model.py
|   └── multi_task_model.py
|
|-- trainer - trainers
|   |-- __init__.py
|   |-- multi_task_trainer.py
|   |-- one_task_trainer.py
|   └── trainer.py
|
|-- utils  - small utility functions
|   |-- __init __.py
|   |-- copy_files.py
|   |-- data_util.py
|   └── util.py
|
|-- logger - module for tensorboard visualization and logging
|   |-- __init __.py
|   |-- logger.py
|   └── logger_config.json
|
|-- docker - contains Dockerfile for containerization of the project
|   └── Dockerfile
|
|-- data
|
|-- saved
|
└── wandb
```

<br></br>
## 💻실행 방법 <a name = 'Code'></a>
- Package install
  ```
  pip install -r requirements.txt
  ```


- Model training
  ```
  python train.py -c train_config.json
  ```
  - train_config.json 
    ```
    {
        "name": "PROJECT_NAME",
        "n_gpu": GPU_NUM(int),
    
        "arch": {
            "type": "MODEL_OBJECT_NAME",
            "args": {}
        },
        "train_set": {
            "type": "DATASET_OBJECT_NAME",
            "args": {
                "data_dir": "TRAIN_SET_DIR",
                "do_calc": CALC_MEAN_STD(boolean),
                "use_all_data": CALC_WITH_ALL_DATA(boolean)
            },
            "use_kfold": TRAIN_WITH_KFOLD(boolean),
            "k_splits": FOLD_NUM(int)
        },
        "valid_set": {
            "type": "DATASET_OBJECT_NAME",
            "args": {
                "data_dir": "VALID_SET_DIR"
            }
        },
        "train_loader": {
            "type": "DATALOADER_OBJECT_NAME",
            "args":{
                "batch_size": (int),
                "shuffle": (boolean),
                "num_workers": (int)
            }
        },
        "valid_loader": {
            "type": "DATALOADER_OBJECT_NAME",
            "args":{
                "batch_size": (int),
                "shuffle": (boolean),
                "num_workers": (int)
            }
        },
        "optimizer": {
            "type": "OPTIMIZER_OBJECT_NAME",
            "args":{
                "lr": (float),
                "weight_decay": (int),
                "amsgrad": (boolean)
            }
        },
        "loss": "LOSS_OBJECT_NAME",
        "metrics": [
            "METRIC_OBJECT_NAME", ...
        ],
        "lr_scheduler": {
            "type": "LR_SCHEDULER_OBJECT_NAME",
            "args": {
                "mode": ("min" or "max"),
                "patience": (int),
                "min_lr": (float)
            }
        },
        "trainer": {
            "type": "TRAINER_OBJECT_NAME",
            "args": {},
    
            "epochs": (int),
    
            "save_dir": "LOG_MODEL_SAVE_DIR",
            "save_period": (int),
            "verbosity": (int),
            
            "monitor": "MODE METRIC",
            "early_stop": (int)
        },
        "wandb": {
            "exp_name": "WANDB_EXP_NAME",
            "exp_num": WANDB_EXP_NUM(int),
            "project_name": "WANDB_PROJECT_NAME",
            "entity": "WANDB_ENTITY_NAME"
        }
    }
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
  - test_config.json
    ```
    {
      "image_path": "TEST_SET_DIR",
      "info_path": "TEST_SET_CSV_PATH",
      "output_path": "RESULT_SAVE_PATH",
      "resize": IMAGE_RESIZE(int),
      "single_model": { # with inference.py (FULL_PATH=saved_dir/saved_exp_name/saved_exp_num/saved_model)
          "saved_dir": "SAVED_MODEL_DIR/WANDB_PROJECT_NAME",
          "saved_exp_name": "WANDB_EXP_NAME",
          "saved_exp_num": WANDB_EXP_NUM(int),
          "saved_model": "SAVED_MODEL.pth",
          "is_multi_task": OUT_FEATURES_(8/18)_CLASSES(true/false)
      },
      "multi_model": { # with ensemble.py
          "saved_dir": "SAVED_MODELS_DIR",
          "saved_models": [
              "SAVED_MODEL.pth", ...
          ],
          "is_multi_task": OUT_FEATURES_(8/18)_CLASSES(true/false),
          "is_deep_model": MULTI_MODEL_PER_TASK_IN_8_CLASSES(boolean)
      }
    }
    ```
