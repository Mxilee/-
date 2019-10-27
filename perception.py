import numpy as np
import matplotlib.pyplot as plt
import time
import random

def createdata():
    samples = np.array([[3, -3], [4, -3], [1, 1], [1, 2],[1,5],[2,5],[6,1],[8,0]])
    labels = [-1, -1, 1, 1, 1,1,-1,-1]
    return samples, labels


def create_line_data(positive_num,negative_num):
    negative_x1 = np.random.uniform(low=-4.0,high= 3,size=(positive_num,1))
    negative_x2 = np.random.uniform(low=-4.0,high=-1,size=(positive_num,1))
    negative_x = np.column_stack((negative_x1,negative_x2))

    positive_x1 = np.random.uniform(low = 0,high = 4.5 ,size =(negative_num,1))
    positive_x2 = np.random.uniform(low = 0,high=4.5,size = (negative_num,1))
    positive_x  = np.column_stack((positive_x1,positive_x2))
    # u = np.mean(samples,axis=0)
    #     # v = np.std(samples,axis =0)
    #     # samples = (samples-u)/v
    #     # print(samples)

    samples = np.row_stack((negative_x,positive_x))

    labels = np.zeros((positive_num+negative_num,1))
    labels[0:positive_num,:] = 1
    labels[positive_num:,:] = -1
    return samples, labels


def create_nonline_data(positive_num,negative_num):
    negative_x1 = np.random.uniform(low=-4.0, high=3, size=(positive_num, 1))
    negative_x2 = np.random.uniform(low= 0 , high= 3 , size=(positive_num, 1))
    negative_x = np.column_stack((negative_x1, negative_x2))

    positive_x1 = np.random.uniform(low= -3, high=0, size=(negative_num, 1))
    positive_x2 = np.random.uniform(low=-3, high= 1, size=(negative_num, 1))
    positive_x = np.column_stack((positive_x1, positive_x2))
    # u = np.mean(samples,axis=0)
    #     # v = np.std(samples,axis =0)
    #     # samples = (samples-u)/v
    #     # print(samples)

    samples = np.row_stack((negative_x, positive_x))

    labels = np.zeros((positive_num + negative_num, 1))
    labels[0:positive_num, :] = 1
    labels[positive_num:, :] = -1
    return samples, labels
# def create_nonline_data(positive_num,negative_num):
#     samples_x1 = np.random.uniform(low=0.0,high=5.0,size=(positive_num,1))
#     samples_x2 = np.random.uniform(low=-5.0,high=5.0,size=(positive_num,1))
#
#     samples = np.column_stack((samples_x1,samples_x2))





class Perception:
    def __init__(self, x, y, max_epochs):
        self.x = x
        self.y = y
        self.max_epochs = max_epochs
        self.w = np.random.randn(x.shape[1])
        self.b = 0
    def sign(self,x,w,b):
        tmpy = np.dot(x,w)+b
        tmpy = 1 if(tmpy>0) else -1
        return tmpy

    def train(self):
        max_epochs = self.max_epochs
        count = 0
        OK = False
        epochs = 0
        while(not OK and epochs<= max_epochs):
            for i in range(self.x.shape[0]):
                y = self.sign(self.x[i],self.w,self.b)
                if(y * self.y[i] <0):
                    self.w = (self.w + self.x[i]*self.y[i])
                    #print(self.w.shape)#.reshape(self.x.shape[1],1)
                    self.b = self.b + self.y[i]
                    break
                else:
                    count += 1
                    if (count == self.x.shape[0]):
                        OK = True
                        print(epochs)
            epochs += 1

        return self.w,self.b


class Pocket_Perception:
    def __init__(self, x, y, max_epochs):
        self.x = x
        self.y = y
        self.b = 0
        self.max_epochs = max_epochs
        self.counter = 0
        self.w = np.random.randn(x.shape[1])

    def sign(self, x, w, b):
        tmpy = np.dot(x, w) + b
        tmpy = 1 if (tmpy > 0) else -1
        return tmpy


    # def count_wrong_number(self,w,b,x,y):
    #     num_wrong = 0
    #     for i in range(x.shape[0]):                                     #先统计初始w下的错误点数,并保存错误列表
    #         tmpy = self.sign(x[i], w, b)
    #         if(tmpy * y[i] <0):
    #             num_wrong += 1
    #     return num_wrong,




    # def train(self):
    #     max_epochs = self.max_epochs
    #     wrong_count = self.count_wrong_number(self.w,self.b,self.x,self.y)
    #     epochs = 0
    #
    #     while(epochs<max_epochs):
    #         for i in range(self.x.shape[0]):
    #             y = self.sign(self.x[i], self.w, self.b)
    #             if (y * self.y[i] < 0):
    #                 tmp_w = (self.w + self.x[i]*self.y[i])
    #                 tmp_b = self.b + self.y[i]
    #                 new_wrong_count,wrong_list = self.count_wrong_number(tmp_w,tmp_b,self.x,self.y)
    #                 if(new_wrong_count <= wrong_count):
    #                     self.w = tmp_w
    #                     self.b = tmp_b
    #                     wrong_count = new_wrong_count
    #                 else:
    #                     continue
    #         epochs+=1
    #     acc = (self.x.shape[0]-wrong_count)/self.x.shape[0]
    #     return self.w, self.b,acc










                                                                            #更新 没有ok 小于max_epochs就循环




    def count_wrong_number(self,w,b,x,y):
        wrong_list = []
        num_wrong = 0
        for i in range(x.shape[0]):                                     #先统计初始w下的错误点数,并保存错误列表
            tmpy = self.sign(x[i], w, b)
            if(tmpy * y[i] <0):
                num_wrong += 1
                wrong_list.append((x[i],i))
        return num_wrong,wrong_list




    def train(self):
        max_epochs = self.max_epochs
        wrong_count,wrong_list = self.count_wrong_number(self.w,self.b,self.x,self.y)
        epochs = 0
        OK = False

        while(not OK and epochs<max_epochs):
            if(len(wrong_list)==0):
                OK = True
            else:
                test_x, i = random.choice(wrong_list)
                tmp_w = (self.w + test_x * self.y[i])
                tmp_b = self.b + self.y[i]
                new_wrong_count, new_wrong_list = self.count_wrong_number(tmp_w, tmp_b, self.x, self.y)
                if (new_wrong_count <= wrong_count):
                    self.w = tmp_w
                    self.b = tmp_b
                    wrong_count = new_wrong_count
                    wrong_list = new_wrong_list

            epochs += 1
        acc = (self.x.shape[0]-wrong_count)/self.x.shape[0]
        return self.w, self.b,acc










class Picture:
    def __init__(self, data, w, b,label,color,name):
        self.b = b
        self.w = w
        plt.figure(1)
        plt.title('Result', size=14)
        plt.xlabel('x0', size=10)
        plt.ylabel('x1', size=10)

        xData = np.linspace(-5, 5, 100)
        yData = self.expression(xData)
        plt.plot(xData, yData, color = color,label=name)

        for i in range(len(data)):
            if (label[i] == 1):
                plt.scatter(data[i][0], data[i][1], c="green", s=50, marker="o")
            else:
                plt.scatter(data[i][0], data[i][1], c="red", s=50, marker="x")
        plt.legend()
        plt.savefig('2d.png', dpi=75)

    def expression(self, x):
        y = (-self.b - self.w[0] * x) / self.w[1]  # 注意在此，把x0，x1当做两个坐标轴，把x1当做自变量，x2为因变量
        return y

    def Show(self):
        plt.show()




if __name__ == '__main__':
    #x, y = create_line_data(,180)
    x,y = create_nonline_data(100,100)
    tic = time.time()
    myperception = Perception(x,y,max_epochs=8000)
    weights, bias = myperception.train()
    #Picture1 = Picture(x, weights, bias, y,"red","PLA")
    toc = time.time()
    print("PLA time cost:")
    print(toc-tic)
    print("-----------------------------------")

    tic1 = time.time()
    my_pocket_perception = Pocket_Perception(x,y,max_epochs=3000)
    weights1,bias1,acc = my_pocket_perception.train()
    tic2 = time.time()
    print("Pocket PLA time cost:")
    print(tic2-tic1)

    Picture2 = Picture(x, weights1, bias1,y,"blue","Pocket_PLA")
    print("The accuracy of Pocket PLA:"+str(acc))


    #pic = plt.figure()
    #plt.subplot(2,1,1)
    #Picture1.Show()
    #plt.subplot(2,1,2)
    Picture2.Show()
