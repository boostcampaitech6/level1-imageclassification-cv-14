# How To use Offline Data Augmentation

```
${data_aug_utils}
├── aug_config.json
├── aug_mask.py
├── aug_age.py
├── rm_back.py
└── README.md
```

1. aug_mask.py
    - aug_config.json의 mask 값 안에서 적용시킬 transfoms, 증강시킬 class, src_dir(원본 데이터 위치), dest_dir(새로 만들 폴더 위치) 수정
    - level1-imageclassification-cv-14 에서 다음 코드 실행
```
python3 ./data_aug_utils/aug_mask.py
```

2. aug_age.py
    - aug_config.json의 age 값 안에서 적용시킬 transfoms, 증강시킬 class, src_dir(원본 데이터 위치), dest_dir(새로 만들 폴더 위치) 수정
    - level1-imageclassification-cv-14 에서 다음 코드 실행
```
python3 ./data_aug_utils/aug_mask.py
```

3. rm_back.py
    - eval 데이터에서 누끼를 제거할 수 있습니다.
    - level1-imageclassification-cv-14 에서 다음 코드 실행
```
python3 ./data_aug_utils/rm_back.py -s {src 폴더 경로} -d {dest 폴더 경로}
```

- 적용 가능한 transform 종류
    - blur : 이미지 흐리게 처리
    - flip : 이미지 좌우 반전
    - jitter : 이미지 랜덤한 명암, 대조 적용
    - clahe : 이미지 clahe 적용
    - rmback : 이미지 배경 제거

- example
    - aug_mask : incorrect, mask, normal 클래스에 대해 배경을 제거한 이미지를 whiht_backs 폴더에 저장
    - aug_age : old 클래스에 대해 blur, flip, jitter 처리를 한 이미지 세트와 flip, jitter 처리를 한 3 개의 이미지 세트를 추가하여 aug_elder 폴더에 저장
```
{
    "mask": {
        "transforms": [
            ["remove_back"]
        ],
        "class":{
            "incorrect" : true,
            "mask" : true,
            "normal" : true
        },
        "src_dir": "./data/train/images",
        "dest_dir": "./data/train/whiht_backs"
    },
    "age": {
        "transforms": [
            ["blur","flip","jitter"],
            ["flip", "jitter"],
            ["flip", "jitter"],
            ["flip", "jitter"]
        ],
        "class":{
            "young" : false,
            "midle" : false,
            "old" : true
        },
        "src_dir": "./data/train/images",
        "dest_dir": "./data/train/aug_elder"
    }
  }

```