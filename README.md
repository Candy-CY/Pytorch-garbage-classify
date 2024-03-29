# Pytorch-garbage-classify  
本科毕业设计：基于深度网络的垃圾识别与分类算法研究  

在现代社会生活与生产活动下，不可避免的会产生巨量且多样的垃圾。我国的人口和经济总量均位居世界前列，因此，必然面临着庞大数量的垃圾处理的难题。如何通过人工智能来对垃圾进行有效分类，成为当前备受关注的研究热点。本文为展开基于深度网络的垃圾识别与分类算法研究，先使用PyTorch框架中的transforms方法对数据进行预处理操作，后经过多次调参实验，对比朴素贝叶斯模型、Keras卷积神经网络模型、ResNeXt101模型的垃圾分类效果。确定最佳分类模型是ResNeXt101，该模型在GPU环境下的分类准确率达到了94.7%。最后利用postman软件来测试API接口，完成图片的在线预测。在微信开发者工具的基础上，利用一些天行数据的垃圾分类的API接口再结合最佳模型的API接口，开发出了一个垃圾分类微信小程序。本文的研究内容丰富和完善了垃圾图像分类的相关研究，也为后续的研究提供了一定的参考价值。  

如遇问题，可联系作者Candy邮箱：candy_2000_0108@163.com 👩‍💻
