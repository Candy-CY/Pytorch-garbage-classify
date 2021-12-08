# -*- coding:utf-8 -*-
# 导入库
import torch
import torch.nn as nn
from torchvision import transforms
# 导入我们自定义的model
import models
#加载模型
model_ft = models.resnext101_32x16d_wsl()
r = model_ft.eval() # 预测问题，指定eval
#print(model_ft)  #模型参数
#加载图片数据
file_name = 'images/yindu.jpg'
from PIL import Image
input_image = Image.open(file_name)
print(input_image)
print(input_image.size)
#图片数据预处理
# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
input_image = Image.open(file_name)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
print('input_tensor.shape = ',input_tensor.shape)
print('input_tensor = ',input_tensor)
# 转化格式
# torch.Size([3, 224, 224])->torch.Size([1, 3, 224, 224])
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
print('convert data format input_batch =  ',input_batch.shape)
#模型在线预测
# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model_ft.to('cuda')
with torch.no_grad():
    output = model_ft(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
print(output[0].shape)
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
print(torch.nn.functional.softmax(output[0], dim=0))
#获取最大可能性的类别：需要获取的数据： id－name 的标签 ，获取结果中最大可能性id号
result = torch.nn.functional.softmax(output[0], dim=0)
# result -> list
v_list = result.cpu().numpy().tolist()
v_max = 0
idx = 0
for i,v in enumerate(v_list):
    if v>v_max:
        v_max = v
        idx = i
print('v_max = ',v_max) # 1000 个分类中，idx 对应的可能性
print('idx = ',idx)
import codecs
ImageNet_dict = {}
for line in  codecs.open('data/ImageNet1k_label.txt','r',encoding='utf-8'):
    line = line.strip() # 0: 'tench, Tinca tinca',                             丁鲷(鱼)
    _id = line.split(":")[0]
    _name = line.split(":")[1]
    _name = _name.replace('\xa0',"")
    ImageNet_dict[int(_id)] = _name
print(ImageNet_dict)
print(ImageNet_dict[idx])
