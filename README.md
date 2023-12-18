# [P Stage1] 도담도담


## Project Overview

- COVID-19 확산으로 인해 마스크의 중요성이 대두됨
- 바이러스는 주로 입과 호흡기로 전파되기 때문에 코와 입을 완전히 가리는 올바른 방법으로 마스크를 착용 해야함
- 모든 사람이 마스크를 착용하여 전차 경로를 차단
- 그러나 모든 사람의 마스크 착용 상태를 확인하는 것은 인력적 제약이 따름
 #### => 카메라를 통해 사람의 얼굴 이미지만으로 마스크 착용 여부를 자동으로 판별할 수 있는 시스템의 개발이 필요




## 최종결과


>Public f1 Score : 

>Private f1 Score : 

>최종 순위 : 


## objective
성별, 연령, 마스크 착용 여부에 따라 사진을 총 18개의 클래스로 분류



## DataSet

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

## How to Solve
- 확장성이 높은 pytorch Template에 부스트캠프에서 제공해준 baseline 중 필요한 부분만 이식하여 사용.

### 활용기법

#### ☞ EDA

#### ☞ Augmentation

#### ☞ Class Imbalance & Data Labeling



---
## MEMBER



