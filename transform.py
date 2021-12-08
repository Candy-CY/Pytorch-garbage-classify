import io
import torchvision.transforms as transforms
from PIL import Image

# 数据预处理方法定义
preprocess = transforms.Compose([
    # 1. 图片变换：重置图像的分辨率,图片缩放 256
    transforms.Resize(256),
    # 2. 裁剪： 中心裁剪，给定的size 从中心裁剪
    transforms.CenterCrop(224),
    # 3. 数据归一化[0,1] 除以255
    transforms.ToTensor(),
    # 4. 对数据进行标准化，即减去我们的均值，然后在除以标准差
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def transform_image(img_bytes):
    """
    图片数据-> 数据预处理
    :param img_bytes:
    :return:
    """
    image = Image.open(io.BytesIO(img_bytes))

    image_tensor = preprocess(image)

    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor
