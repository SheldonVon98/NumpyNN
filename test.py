# Copyright (c) SheldonFung All Rights Reserved.
# File Name: test.py
# Author: Sheldon Fung
# email: sheldonvon@outlook.com

import numpy as np

# 设置模型大小
input, hide_1, hide_2, output = 2, 5, 4, 1

# 生成200个数据 
# 每一个数据由两个数字组成，标签为两个数的和。
data = np.random.random(size=(200,2))
w = [1, 1]
label = np.matmul(data, w).reshape((200,1))

#100个训练数据
train_data = data[:1]
train_label = label[:1]

#100个测数据
test_data = data[100:]
test_label = data[100:]


#随机生成 w1 和 w2 训练参数
w1 = np.random.randn(input, hide_1)   # (2, 5)
w2 = np.random.randn(hide_1, hide_2)  # (5, 4)
w3 = np.random.randn(hide_2, output)  # (4, 1)
#w1 = np.ones((input, hide_1))   # (2, 5)
#w2 = np.ones((hide_1, hide_2))  # (5, 4)
#w3 = np.ones((hide_2, output))  # (4, 1)


#学习率
learning_rate = 1e-5


#训练
def train(train_data, train_label, input, hide_1, hide_2, output, w1, w2, w3):
    for epoch in range(50000):   
        #前向传播
        #h0层
        h0 = train_data.dot(w1)         # (100, 2) * (2, 5) -> (100, 5)
        h0_relu = np.maximum(h0, 0)  

        #h1层
        h1 = h0_relu.dot(w2)            # (100, 5) * (5, 4) -> (100, 4)
        h1_relu = np.maximum(h1, 0)   

        #输出层
        y_pred = h1_relu.dot(w3)        # (100, 4) * (4, 1) -> (100, 1)

        #损失函数
        loss = np.square(y_pred - train_label).sum()

        #反向传播 求出w1,w2,w3的梯度
        #输出层
        grad_y_pred = 2.0 * (y_pred - train_label)

        # h1层


        #print(h1_relu, end="\n\n")
        #print(grad_y_pred, end="\n\n")
        grad_w3 = h1_relu.T.dot(grad_y_pred)

        #print(grad_w3, end="\n\n")
        grad_h1_relu = grad_y_pred.dot(w3.T) 
        #print(grad_h1_relu, end="\n\n")
        grad_h1 = grad_h1_relu.copy()
        #print(grad_h1, end="\n\n")
        grad_h1[h1<0] = 0
        #exit(0)
        #h0层
        #print(h0_relu, end="\n\n")
        print(grad_h1, end="\n\n")
        print(w2, end="\n\n")
        grad_w2 = h0_relu.T.dot(grad_h1)
        #print(grad_w2)
        grad_h0_relu = grad_h1.dot(w2.T) 
        print(grad_h0_relu)
        exit(0)
        grad_h0 = grad_h0_relu.copy()
        grad_h0[h0<0] = 0
        #exit(0)
        #输入层
        grad_w1 = train_data.T.dot(grad_h0)

        # 更新参数 w1 和 w2
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2
        w3 -= learning_rate * grad_w3
        if epoch % 10000 == 0:
            #print("train",w1)
            print("train_loss", loss/100)      
try:
    train(train_data, train_label, input, hide_1, hide_2, output, w1, w2, w3) 
except KeyboardInterrupt:
    pass

#测试
def test(test_data, input, hide_1, hide_2, output, w1, w2, w3):
    h0 = test_data.dot(w1)          
    h0_relu = np.maximum(h0, 0)  
#
    #h1层
    h1 = h0_relu.dot(w2)            
    h1_relu = np.maximum(h1, 0)   

    #输出层
    y_pred = h1_relu.dot(w3)        
    #损失函数
    print("")
    print(y_pred)

test_data = np.array([[0.1, 0.5 ]])                            # 0.1和0.5 可以修改
test(test_data, input, hide_1, hide_2, output, w1, w2, w3)
