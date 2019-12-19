import random
import torch
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader

def datafn(expnum=10):
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

    data=random.sample(range(0,31),expnum)
    while (15 in data) or (20 in data):
        if 15 in data:
            data.remove(15)
            data = data + random.sample(range(0, 31), 1)
        if 20 in data:
            data.remove(20)
            data = data + random.sample(range(0, 31), 1)
    input=[]
    output=[]
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
        input.append(bin)
        output.append(four)
    return input,output


class Trainer(object):

    def __init__(self, model=None, criterion=None, optimizer=None, dataset=None, USE_CUDA=False):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataset = dataset
        self.iterations = 0
        self.USE_CUDA = USE_CUDA

    def run(self, epochs=1):
        # 每一个epoch 就是一次train的过程
        for i in range(1, epochs + 1):
            self.train()
        self.test()

    def train(self):
        # 从dataloader 中拿数据
        for i, data in enumerate(self.dataset, self.iterations + 1):
            batch_input, batch_target = data
            input_var = batch_input
            target_var = batch_target
            if self.USE_CUDA:
                input_var = input_var.cuda()
                target_var = target_var.cuda()

            # 每一次前馈就是一次函数闭包操作
            def closure():
                batch_output = self.model(input_var)
                print("input_var:{} batch_output:{} batch_target:{}".format(input_var,batch_output,batch_target))
                loss = self.criterion(batch_output, target_var)
                loss.backward()
                return loss

            # loss 返回,准备优化
            self.optimizer.zero_grad()
            self.optimizer.step(closure)
        self.iterations += i

    def test(self):
        test_input1=[0,1,1,1,1]
        test_input1=torch.Tensor(test_input1)
        test_output1 = self.model(test_input1)
        test_input2 = [1, 0, 1, 0, 0]
        test_input2 = torch.Tensor(test_input2)
        test_output2 = self.model(test_input2)
        print("测试如下：")
        print("test_input1:{} test_output1:{}".format(test_input1,test_output1))
        print("test_input2:{} test_output2:{}".format(test_input2,test_output2))




class MyLayer(torch.nn.Module):
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

D_in,D_out=5,3

class MyNet(torch.nn.Module):

    def __init__(self):
        super(MyNet, self).__init__()  # 第一句话，调用父类的构造函数
        self.mylayer1 = MyLayer(D_in, D_out)

    def forward(self, x):
        x = self.mylayer1(x)
        return x

class SubDataset(Dataset.Dataset):
    def __init__(self, Data, Label):
        self.Data = Data

        self.Label = Label

    # 返回数据集大小

    def __len__(self):
        return len(self.Data)

    # 得到数据内容和标签

    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index])

        label = torch.Tensor(self.Label[index])

        return data, label


def builder_trainer():
    model=MyNet()
    criterion=torch.nn.MSELoss(reduction='sum')
    learning_rate = 1e-4
    optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)
    DTinput,DToutput=datafn()
    print("数据集如下所示：")
    print(DTinput)
    print(DToutput)
    print("\n\n")
    data=SubDataset(DTinput,DToutput)
    dataloader=DataLoader.DataLoader(data,batch_size=1,shuffle=False)
    trainer=Trainer(model=model,criterion=criterion,optimizer=optimizer,dataset=dataloader)
    return trainer


if __name__=='__main__':
    trainer=builder_trainer()
    trainer.run(1)