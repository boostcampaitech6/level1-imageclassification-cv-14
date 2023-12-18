from enum import Enum
from typing import Tuple
import numpy as np
import torch

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]

def is_image_file(filename):
    """파일 이름이 이미지 확장자를 가지는지 확인하는 함수

    Args:
        filename (str): 확인하고자 하는 파일 이름

    Returns:
        bool: 파일 이름이 이미지 확장자를 가지면 True, 그렇지 않으면 False.
    """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class MaskLabels(int, Enum):
    """마스크 라벨을 나타내는 Enum 클래스"""

    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    """성별 라벨을 나타내는 Enum 클래스"""

    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        """문자열로부터 해당하는 성별 라벨을 찾아 반환하는 클래스 메서드"""
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(
                f"Gender value should be either 'male' or 'female', {value}"
            )


class AgeLabels(int, Enum):
    """나이 라벨을 나타내는 Enum 클래스"""

    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        """숫자로부터 해당하는 나이 라벨을 찾아 반환하는 클래스 메서드"""
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 60:
            return cls.MIDDLE
        else:
            return cls.OLD
        

def encode_multi_class(mask_label, gender_label, age_label) -> int:
    """다중 라벨을 하나의 클래스로 인코딩하는 메서드"""
    return mask_label * 6 + gender_label * 3 + age_label


def decode_multi_class(multi_class_label) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
    """인코딩된 다중 라벨을 각각의 라벨로 디코딩하는 메서드"""
    mask_label = torch.div(multi_class_label, 6, rounding_mode='floor')
    gender_label = torch.div(multi_class_label, 3, rounding_mode='floor') % 2
    age_label = multi_class_label % 3
    return mask_label, gender_label, age_label


def denormalize_image(image, mean, std):
    """정규화된 이미지를 원래대로 되돌리는 메서드"""
    img_cp = image.copy()
    img_cp *= std
    img_cp += mean
    img_cp *= 255.0
    img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
    return img_cp