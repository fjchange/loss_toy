# loss_toy
toy model to visualize feature space of models trained by different loss functions

目的是可视化特征空间，然后选择一种loss 函数 尝试训练我们自己的数据集。


已经完成 Triplet Loss 与 Center Loss 的可视化
期望完成：
1. Large-Margin Softmax Loss [论文连接](http://proceedings.mlr.press/v48/liud16.pdf)
2. Angular-Softmax Loss   [论文链接](https://arxiv.org/abs/1704.08063)
3. Triplet-Center Loss [论文链接](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1632.pdf)
4. Margin Loss [论文链接](https://arxiv.org/pdf/1706.07567.pdf)

Tensorflow自带的读入不一定满足模型的需求，所以mnist_img_train就是做把数据还原成图片的，然后放好，可以根据模型需要自己来改输入。

模型就是最终利用的可视化的是倒数第二层FC的二维向量来可视化的。

希望测试的是open set下的可视化效果，所以训练的时候是9个数字，测试的时候是完整的，具体vis.py写好了。
