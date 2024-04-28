import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output
#load datashet
print("Load MNIST Database")
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data() #lưu vào tập train và test
x_train=np.reshape(x_train,(60000,784))/255.0 #chia 255 là để chuẩn hóa 0,1
x_test=np.reshape(x_test,(10000,784))/255.0
y_train=np.matrix(np.eye(10)[y_train])
y_test=np.matrix(np.eye(10)[y_test])
print("------------------------")
print(x_train.shape)
print(y_train.shape)

# activation function - hàm kích hoạt
# 1st hidden layer
def relu(x):
    return np.where(x > 0, 1, 0)
# 2nd hidden layer
def sigmoid(x):
    return 1. / (1. + np.exp(-x))
# output layer
def softmax(x):
  return np.divide(np.matrix(np.exp(x)),np.mat(np.sum(np.exp(x),axis=1)))

# Feedforward propagation - lan truyền tới
def Forwardpass(X,Wh2,bh2,Wh1,bh1,Wo,bo):
  zh1=X@Wh1.T + bh1
  a1=relu(zh1)
  zh2=a1@Wh2.T + bh2
  a2=sigmoid(zh2)
  z=a2@Wo.T + bo
  o=softmax(z)
  return o
def AccTest(label,prediction):
  OutMaxArg=np.argmax(prediction,axis=1)
  LabelMaxArg=np.argmax(label,axis=1)
  Accuracy=np.mean(OutMaxArg==LabelMaxArg)
  return Accuracy

#tốc độ học và các thông số inputlayer,hidden layer, output layer,...
learningRate=0.001
Epoch=350 #xong vongf lặp thì được 1 epoch
NumTrainSamples=60000
NumTestSamples=10000

NumInputs=784
NumHiddenUnits=512
NumClasses=10

#khởi tạo weigts ban đầu ở:
#1st hidden layer
Wh1=np.matrix(np.random.uniform(-0.5,0.5,(NumHiddenUnits,NumInputs)))# node ở hidden layer và ngõ vào ảnh 28x28
bh1=np.random.uniform(0,0.5,(1,NumHiddenUnits))
dWh1= np.zeros((NumHiddenUnits,NumInputs))#tạo các ma trận giá trị 0
dbh1=np.zeros((1,NumHiddenUnits))
# 2nd hidden layer
Wh2=np.matrix(np.random.uniform(-0.5,0.5,(NumHiddenUnits,NumHiddenUnits)))#tầng 2 lớp layer node:512 nên tạo wh2 (512,512) để cùng shape
bh2=np.random.uniform(0,0.5,(1,NumHiddenUnits))
dWh2= np.zeros((NumHiddenUnits,NumInputs))
dbh2=np.zeros((1,NumHiddenUnits))
#output layer
Wo=np.random.uniform(-0.5,0.5,(NumClasses,NumHiddenUnits))#ngõ ra là 10 node
bo=np.random.uniform(0,0.5,(1, NumClasses))
dWo=np.zeros((NumClasses,NumHiddenUnits))
dbo=np.zeros((1,NumClasses))


loss=[]
Acc=[]
Batch_size=200 # theo đề bài 200
Stochastic_samples= np.arange(NumTrainSamples)

for ep in range(Epoch):
    np.random.shuffle(Stochastic_samples)
    for ite in range(0, NumTrainSamples, Batch_size):
        # lấy một batch mẫu từ tập dữ liệu để train
        Batch_samples = Stochastic_samples[ite:ite + Batch_size]

        x = x_train[Batch_samples, :]
        y = y_train[Batch_samples, :]

        # các activation function đã tạo
        zh1=x@Wh1.T + bh1
        ah1=relu(zh1)
        zh2=ah1@Wh2.T + bh2
        ah2=sigmoid(zh2)
        z=ah2@Wo.T + bo
        o=softmax(z)

        # tính toán Loss
        loss.append(-np.sum(np.multiply(y, np.log10(o))))

        # Caculate the error for the output laye
        d = o - y

        # Back propagate error
        # đã chứng minh công thức ở câu 1: (o-y)Wo
        dh = d @ Wo

        # (o-y).Wo.ah2(1-ah2) dùng multiply nhân phần tử 
        dhs2= np.multiply(np.multiply(dh, ah2), (1 - ah2))
        
        # ah1 có giá trị là = 0 và ( zh1 nếu zh1 > 0 ) 
        # đạo hàm của ah1 có 2 trường hợp 
        # tạo ma trận new_matrix có ý nghĩa so sánh từng giá trị của ma trận và trả về theo điều kiện 
        # khi ah1 > 0 thì lúc này new_matrix là 1 ( tương ứng với khi ah1 = zh1 và đạo hàm zh1 = 1 ) 
        new_matrix = np.where(ah1 > 0, 1, 0)

        # dùng multiply nhân phần tử còn @Wh2 nhân ma trận vì Wh2 là trọng số
        dhs1 = np.multiply(np.multiply(np.multiply(dh, ah2), (1 - ah2))@ Wh2, new_matrix)

        # tính toán đạo hàm riêng tại cái layer
        dWo = np.matmul(np.transpose(d), ah2) #matul là phép nhân, trước khi nhân phải chuyển vị d
        dbo = np.mean(d) 
        dWh2 = np.matmul(np.transpose(dhs2), ah1)
        dbh2 = np.mean(dhs2)  
        dWh1 = np.matmul(np.transpose(dhs1), x)
        dbh1 = np.mean(dhs1)  
        
        # cập nhật trọng số khi lấy đạo hàm chia cho batch_size
        Wo = Wo - learningRate * dWo / Batch_size
        bo = bo - learningRate * dbo
        Wh2 = Wh2 - learningRate * dWh2 / Batch_size
        bh2 = bh2 - learningRate * dbh2
        Wh1= Wh1-learningRate*dWh1 / Batch_size
        bh1 = bh1 - learningRate * dbh1

        

    # Test accuracy with random initial weights
    prediction = Forwardpass(x_test,Wh2,bh2,Wh1,bh1,Wo,bo)
    Acc.append(AccTest(y_test, prediction))

    clear_output(wait=True)
    plt.plot([i for i, _ in enumerate(Acc)], Acc, 'o')


    prediction = Forwardpass(x_test,Wh2,bh2,Wh1,bh1,Wo,bo)
    Rate = AccTest(y_test, prediction)
    # in ra các rate của mỗi epoch
    print(Rate)
    
plt.show() # hiển thị biểu đồ sau khi hoàn thành tất cả các epoch