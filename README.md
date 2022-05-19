# DeepLearningUtil
딥러닝에 사용했던 유틸리티 함수들을 저장하는 문서

## dataset_utils
dataset에 관련된 것들

#### get_dataset.ipynb:
Cellphone Segmentation Set Download Example 작성
1. 데이터셋 다운로드 툴인 fiftyone으로 Open Image Dataset v6+를 다운로드(Segmentation)
2. 원하는 class 데이터만 가져다 image,mask 크기를 맞추고 배경 분리 등 가시화/저장하는 코드.(Segmentation)

#### browse_datasets.ipynb
Fiftyone으로 부터 다운로드 받은 데이터를 확인
1. fiftyone으로 부터 다운로드 받은 이미지, 바운딩 박스 가시화 코드
2. 필터링&저장 인터페이스 (사용용도: 데이터 셋의 범위가 원하는 것보다 넓어 필요한 양질의 데이터를 필터링해야 할 때 사용)

#### split_dataset.ipynb
train, validation set으로 나누는 코드. 범용성이 낮아 수정 필요있음. (이미지와 텍스트의 이름과 개수가 일치해야만 제대로 작동함)



## augment_transform
Augmentation과 Transform에 관련된 것들

### copy_paste_aug.ipynb
copy-paste augmentation을 구현. 아직은 범용성이 낮지만, 인수를 받아 여러 선택작업을 할 수 있게 하는 것이 목표

* label은 bounding box(detection)를 뜻하고, mask는 segmentation을 뜻함
* 다음의 경우로 나눠서 사용할 수 있도록 구현하는 것이 목표. 베이스 이미지를 home, 붙일 이미지를 away라 표현함.

1. 배경이미지에 특정 이미지를 붙임
* 필요요소: {home: {image}, away: {image, mask}}
* 결과물: image, label (mask의 경우 기존과 동일)
2. bbox를 가진 배경이미지에 특정 이미지를 붙임
* 필요요소: {home: {image, label}, away: {image, mask}]
* 결과물: image, label (이 경우에 mask는 사용할 수 없음)
3. (미구현) mask를 가진 배경이미지에 특정 이미지를 붙임
* 필요요소: {home: {image, mask}, away: {image, mask}]

추가로 데이터 포멧에 맞춰 label을 생성하고자 함
1. yolo format(center xywh)
2. (미구현) pascal voc format(xyxy)
3. (미구현) coco format(xywh)
* (참고) open image dataset은 특이하게도 xxyy로 되어있음

#### custom_transforms.py
U2Net기반으로 사용했던 custom transform 함수들을 작성해놓은 코드