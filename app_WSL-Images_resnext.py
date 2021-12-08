from flask import Flask,request,jsonify

import torch
from transform import transform_image
import models
from collections import OrderedDict
import time
app = Flask(__name__)

import codecs

# id-> name mapping
ImageNet_dict = {}
for line in codecs.open('data/ImageNet1k_label.txt', 'r',encoding='utf-8'):
    line = line.strip()  # 0: 'tench, Tinca tinca',                             丁鲷(鱼)

    _id = line.split(":")[0]
    _name = line.split(":")[1]
    _name = _name.replace('\xa0', "")
    ImageNet_dict[int(_id)] = _name

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Pytorch facebookresearch_WS-Images_resnext predict device = ',device)

# 加载模型
model_ft = models.resnext101_32x16d_wsl()
model_ft.to(device)
model_ft.eval() # 预测问题，指定eval

@app.route('/')
def hello():
    return "Hello World"


@app.route('/predict', methods=['POST'])
def predict():
    # 获取输入数据
    file = request.files['file']

    img_bytes = file.read()

    # 数据预处理
    image_tensor  = transform_image(img_bytes = img_bytes)
    image_tensor = image_tensor.to(device)

    # 模型预测
    # move the input and model to GPU for speed if available
    input_batch = image_tensor.to(device)
    with torch.no_grad():
        t1 = time.time()
        output = model_ft(input_batch)
        t2 = time.time()
        consume = (t2 - t1) * 1000
        consume = int(consume)

    outputs = torch.nn.functional.softmax(output[0], dim=0)
    # result -> list
    pred_list = outputs.cpu().numpy().tolist()

    # API 接口封装
    label_c_mapping = {}
    for i,prob in enumerate(pred_list):
        label_c_mapping[i] = prob

    ## 获取topK=5 数据结果,按照dict 中prob 进行排序
    topK = 5
    data_list = []
    for label_prob in sorted(label_c_mapping.items(),key=lambda x:x[1],reverse=True)[:topK]:
        label = int(label_prob[0])
        result = {'label':label,'prob':label_prob[1],'name':ImageNet_dict[label]}

        data_list.append(result)
    # JSON 格式数据输出
    result = OrderedDict(error = 0,errmsg = 'success',consume = consume,data=data_list)

    return jsonify(result)

if __name__ == '__main__':
    app.run()
