# 1. 导入库
## 系统库
import os
from os import walk
## torch 相关的库包
import torch
import torch.nn as nn
from torchvision import datasets
## 相关参数
from args import args
## 数据预处理函数定义
from transform import preprocess
## 模型pre_trained_model 加载、训练、评估、标签映射关系
from model import train, evaluate, initital_model, class_id2name
## 工具类： 日志类工具、模型保存、优化器
from utils.logger import Logger
from utils.misc import save_checkpoint, get_optimizer
## 训练矩阵效果评估工具类
from sklearn import metrics
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 5. 定义模型训练和验证方法
def run(model,train_loader,val_loader):

    '''
    模型训练和预测
    :param model: 初始化的model
    :param train_loader: 训练数据
    :param val_loader: 验证数据
    :return:
    '''

    # 初始化变量
    ## 模型保存的变量
    global best_acc
    ## 训练C类别的分类问题，我们CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    ## torch.optim 是一个各种优化算法库
    ## optimizer 对象能保存当前的参数状态并且基于计算梯度更新参数
    optimizer = get_optimizer(model,args)

    #加载checkpoint： 可以指定迭代的开始位置进行重新训练
    if args.resume:
        # --resume checkpoint/checkpoint.pth.tar
        # load checkpoint
        print('Resuming from checkpoint...')
        assert os.path.isfile(args.resume),'Error: no checkpoint directory found!!'
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        state['start_epoch'] = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    #评估: 混淆矩阵；准确率、召回率、F1-score
    if args.evaluate:
        print('\nEvaluate only')
        test_loss, test_acc, predict_all,labels_all = evaluate(val_loader,model,criterion,test=True)
        print('Test Loss:%.8f,Test Acc:%.2f' %(test_loss,test_acc))

        # 混淆矩阵
        report = metrics.classification_report(labels_all,predict_all,target_names=class_list,digits=4)
        confusion = metrics.confusion_matrix(labels_all,predict_all)

        print('\n report ',report)
        print('\n confusion',confusion)
        return
    #模型的训练和验证
    ## append logger file
    logger = Logger(os.path.join(args.checkpoint,'log.txt'),title=None)
    ## 设置logger 的头信息
    logger.set_names(['Learning Rate', 'epoch', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
    for epoch in range(state['start_epoch'],state['epochs']+1):
        print('[{}/{}] Training'.format(epoch,args.epochs))
        # train
        train_loss,train_acc = train(train_loader,model,criterion,optimizer)
        # val
        test_loss,test_acc = evaluate(val_loader,model,criterion,test=None)

        # 核心参数保存logger
        logger.append([state['lr'], int(epoch), train_loss, test_loss, train_acc, test_acc])
        print('train_loss:%f, val_loss:%f, train_acc:%f,  val_acc:%f' % (
            train_loss, test_loss, train_acc, test_acc,))
        # 保存模型
        is_best = test_acc > best_acc
        best_acc = max(test_acc,best_acc)
        save_checkpoint({
            'epoch':epoch + 1,
            'state_dict':model.state_dict(),
            'train_acc':train_acc,
            'test_acc':test_acc,
            'best_acc':best_acc,
            'optimizer':optimizer.state_dict()

        }, is_best, checkpoint=args.checkpoint)


    print('Best acc:')
    print(best_acc)

# 入门程序

#if __name__ == '__main__':
if __name__ == '__main__':
    # 1. 获取所有的参数
    state = {k:v for k,v in args._get_kwargs()}
    print('state = ',state)

# 2. 数据整体探测
    base_path = 'data/garbage-classify-for-pytorch'
    for (dirpath, dirnames, filenames) in os.walk(base_path):
            if len(filenames) > 0:
                print('*' * 60)
                print('Diretory path:', dirpath)
                print('total examples = ', len(filenames))
                print('File name Example = ', filenames[:2])

# 3. 数据封装ImageFolder 格式
    TRAIN = "{}/train".format(base_path)
    VALID = "{}/val".format(base_path)
    print('train data_path = ', TRAIN)
    print('val data_path = ', VALID)

##root (string): 根目录路径
##transform: 定义的数据预处理函数
    train_data = datasets.ImageFolder(root=TRAIN, transform=preprocess)
    val_data = datasets.ImageFolder(root=VALID, transform=preprocess)

    assert train_data.class_to_idx.keys() == val_data.class_to_idx.keys()
    print('imgs = ', train_data.imgs[:2])
# 4. 批量数据加载
    batch_size = 10
    num_workers = 2
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    image,label= next(iter(train_loader))
    print(label)
    print(image.shape)
    class_list = [class_id2name()[i] for i in list(range(len(train_data.class_to_idx.keys())))]
    print('class_list = ',class_list)
# 定义全局变量，保存准确率
    best_acc = 0
    print("hello")
    # 模型初始化
    model_name = args.model_name
    num_classes = args.num_classes
    model_ft = initital_model(model_name,num_classes,feature_extract=True)
# 设置模型运行模式（cuda／cpu)
    model_ft.to(device)
# 打印模型参数大小
    print('Total params: %.2fM' % (sum(p.numel() for p in model_ft.parameters()) / 1000000.0))
#print(model_ft)
    run(model_ft,train_loader,val_loader)

