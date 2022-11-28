- ### copy_paste_aug.ipynb
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

- ### custom_transforms.py
U2Net기반으로 사용했던 custom transform 함수들을 작성해놓은 코드