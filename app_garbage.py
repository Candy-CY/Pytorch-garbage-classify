
import torch
from flask import Flask, request
from torch.utils.checkpoint import checkpoint
from utils.json_utils import jsonify
from model import initital_model,class_id2name
from transform import transform_image
import time
from collections import OrderedDict
import codecs
from args import args
# 获取所有配置参数
state = {k: v for k, v in args._get_kwargs()}
print("state = ", state)

app = Flask(__name__)
# 设置编码-否则返回数据中文时候-乱码
app.config['JSON_AS_ASCII'] = False
# 加载Label2Name Mapping
class_id2name = {} 
for line in codecs.open('data/garbage_label.txt', 'r', encoding='utf-8'):
    line = line.strip()
    _id = line.split(":")[0]
    _name = line.split(":")[1]
    class_id2name[int(_id)] = _name

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
print('Pytorch garbage-classification Serving on {} ...'.format(device))
num_classes = len(class_id2name)
model_name = args.model_name
#model_path = args.resume#--resume checkpoint/garbage_resnext101_model_2_1111_4211.pth
model_path ='checkpoint/garbage_resnext101_model_2_2778_2632.pth'
#model_path ='checkpoint/garbage_resnext101_model_5_9233_9474.pth'#'checkpoint/garbage_resnext101_model_2_1667_2105.pth'
print("model_name = ",model_name)
print("model_path = ",model_path)

model_ft = initital_model(model_name, num_classes, feature_extract=True)
model_ft.to(device)  # 设置模型运行环境
# 指定map_location='cpu' ，GPU 环境下训练的模型可以在CPU环境加载并使用[本地测试CPU可以测试，线上环境GPU模型]
checkpoint=torch.load(model_path, map_location='cpu')
model_ft.load_state_dict(checkpoint['state_dict'],False)
model_ft.eval()


@app.route('/')
def hello():
    return "Hello World!"

@app.route('/predict', methods=['POST'])
def predict():
    # 获取输入数据
    file = request.files['file']
    img_bytes = file.read()
    # 特征提取
    feature = transform_image(img_bytes)
    feature = feature.to(device)  # 在device 上进行预测

    # 模型预测
    with torch.no_grad():
        t1 = time.time()
        outputs = model_ft.forward(feature)
        consume = (time.time() - t1) * 1000
        consume = int(consume)

    # API 结果封装
    label_c_mapping = {}
    ## The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    ## 通过softmax 获取每个label的概率
    outputs = torch.nn.functional.softmax(outputs[0], dim=0)
    pred_list = outputs.cpu().numpy().tolist()

    for i, prob in enumerate(pred_list):
        label_c_mapping[int(i)] = prob
    ## 按照prob 降序，获取topK = 5
    dict_list = []
    for label_prob in sorted(label_c_mapping.items(), key=lambda x: x[1], reverse=True)[:5]:
        label = int(label_prob[0])
        result = {'label': label, 'c': label_prob[1], 'name': class_id2name[label]}
        dict_list.append(result)
    ## dict 中的数值按照顺序返回结果
    result = OrderedDict(error=0, errmsg='success', consume=consume, data=dict_list)
    return jsonify(result)


if __name__ == '__main__':
    # curl -X POST -F file=@cat_pic.jpeg http://localhost:5000/predict
    app.run()
