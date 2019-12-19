import torch
import random
# 定义一个 my_layer.py

from torch.autograd import Variable

class MyLayer(torch.nn.Module):
    '''

    因为这个层实现的功能是：y=weights*sqrt(x2+bias),所以有两个参数：

    权值矩阵weights

    偏置矩阵bias

    输入 x 的维度是（in_features,)

    输出 y 的维度是（out_features,) 故而

    bias 的维度是（in_fearures,)，注意这里为什么是in_features,而不是out_features，注意体会这里和Linear层的区别所在

    weights 的维度是（in_features, out_features）注意这里为什么是（in_features, out_features）,而不是（out_features, in_features），注意体会这里和Linear层的区别所在

    '''

    def __init__(self, in_features, out_features, bias=True):
        super(MyLayer, self).__init__()  # 和自定义模型一样，第一句话就是调用父类的构造函数
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.zeros(in_features, out_features))  # 由于weights是可以训练的，所以使用Parameter来定义
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))  # 由于bias是可以训练的，所以使用Parameter来定义
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        # input_ = torch.pow(input, 2) + self.bias
        y = torch.matmul(input, self.weight)+self.bias
        return y


N, D_in, D_out = 10, 5, 3  # 一共10组样本，输入特征为5，输出特征为3


# 先定义一个模型

class MyNet(torch.nn.Module):

    def __init__(self):
        super(MyNet, self).__init__()  # 第一句话，调用父类的构造函数
        self.mylayer1 = MyLayer(D_in, D_out)

    def forward(self, x):
        x = self.mylayer1(x)
        return x

class Trainer():
    pass

def builder_trainer():
    pass

def dataset():
    def to_two(n):
        if n==0:
            return ''
        else:
            return to_two(int(n/2))+str(n%2)
    def to_four(n):
        if n==0:
            return ''
        else:
            return to_four(int(n/4))+str(n%4)

    data=random.sample(range(0,31),10)
    for data_item in data:
        # 转2进制
        bin = to_two(data_item)
        bin = (5 - len(bin)) * '0' + bin
        bin = list(bin)
        bin = list(map(int, bin))

        # 转4进制
        four = to_four(data_item)
        four = (3 - len(four)) * '0' + four
        four = list(four)
        four = list(map(int, four))
        yield data_item,bin,four

def test(model):
    def to_two(n):
        if n == 0:
            return ''
        else:
            return to_two(int(n / 2)) + str(n % 2)

    def to_four(n):
        if n == 0:
            return ''
        else:
            # print("n/4:{} n%4:{}".format(int(n/4),n%4))
            return to_four(int(n / 4)) + str(n % 4)
    data = [15,20]
    for data_item in data:
        bin=to_two(data_item)
        bin = (5 - len(bin)) * '0' + bin
        bin = list(bin)
        bin = list(map(int, bin))
        bin=torch.Tensor(bin)

        four=to_four(data_item)
        four = (3 - len(four)) * '0' + four
        four = list(four)
        four = list(map(int, four))

        y_pred = model(bin)
        y_pred=y_pred.data
        print("真实值：{} 预测值：{}".format(four,y_pred))

def train(model):
    def train_iter():
        def to_two(n):
            if n == 0:
                return ''
            else:
                return to_two(int(n / 2)) + str(n % 2)

        def to_four(n):
            if n == 0:
                return ''
            else:
                # print("n/4:{} n%4:{}".format(int(n/4),n%4))
                return to_four(int(n / 4)) + str(n % 4)

        data = random.sample(range(0, 31), 10)
        while (15 in data) or (20 in data):
            if 15 in data:
                data.remove(15)
                data=data+random.sample(range(0, 31), 1)
            if 20 in data:
                data.remove(20)
                data = data + random.sample(range(0, 31), 1)
        # print("data:".format(data))
        for data_item in data:
            # print("n:{}".format(data_item))
            # 转2进制
            bin = to_two(data_item)
            bin = (5 - len(bin)) * '0' + bin
            bin = list(bin)
            bin = list(map(int, bin))

            # 转4进制
            four = to_four(data_item)
            # print("four:{}".format(four))
            four = (3 - len(four)) * '0' + four
            four = list(four)
            four = list(map(int, four))
            bin,four=torch.Tensor(bin),torch.Tensor(four)
            yield data_item,bin,four

    # train_iter()

    # 创建输入、输出数据
    # x = torch.randn(N, D_in)  # （10，5）
    # y = torch.randn(N, D_out)  # （10，3）

    # 定义损失函数
    loss_fn = torch.nn.MSELoss(reduction='sum')
    learning_rate = 1e-4

    # 构造一个optimizer对象
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    count=0
    for t in range(10000):  #
        # import pdb
        # pdb.set_trace()
        # 第一步：数据的前向传播，计算预测值p_pred
        gener = train_iter()
        for num in range(8):
            dt, x, y = gener.__next__()
            y_pred = model(x)
            # 第二步：计算计算预测值p_pred与真实值的误差
            loss = loss_fn(y_pred, y)
            if count%200==0:
                print("训练数据--dt:{} x:{} y:{}".format(dt, x.data, y.data))
                print(f"第 {count} 个epoch, 损失是 {loss.item()}")
            count=count+1
            # 在反向传播之前，将模型的梯度归零，这
            optimizer.zero_grad()
            # 第三步：反向传播误差
            loss.backward()
            # 直接通过梯度一步到位，更新完整个网络的训练参数
            optimizer.step()



if __name__=='__main__':
    model = MyNet()
    print(model)
    '''运行结果为：

    MyNet(

      (mylayer1): MyLayer()   # 这就是自己定义的一个层

    )

    '''
    train(model)
    test(model)
