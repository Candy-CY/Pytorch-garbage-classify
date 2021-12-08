
# 导入模块
import argparse

# 创建一个参数的解析对象
parser = argparse.ArgumentParser(description='Pytorch garbage Training ')

# 设置参数信息
## 模型名称
parser.add_argument('--model_name', default='resnext101_32x8d', type=str,
                    choices=['resnext101_32x8d', 'resnext101_32x16d'],
                    help='model_name selected in train')

## 学习率

parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initital learning rate 1e-2,12-4,0.001')

## 模型评估 默认false,指定 －e true
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

## 模型的存储路径

parser.add_argument('--resume', default="", type=str, metavar='PATH', help='path to latest checkpoint')

parser.add_argument('-c', '--checkpoint', default="checkpoint", type=str, metavar='PATH',
                    help='path to save checkpoint')

## 模型迭代次数
parser.add_argument('--epochs', default=2, type=int, metavar='N', help='number of total epochs to run')

## 图片分类g
parser.add_argument('--num_classes', default=4, type=int, metavar='N', help='number of classes')

## 从那个epoch 开始训练
parser.add_argument('--start_epoch', default=1, type=int, metavar='N', help='manual epoch number')

# 模型优化器
parser.add_argument('--optimizer', default='adam', choices=['sgd', 'adam'], metavar='N',
                    help='optimizer(default adam)')
# 进行参数解析
args = parser.parse_args()


