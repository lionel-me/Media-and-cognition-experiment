from numpy import*
import numpy
import cv2#opencv
import torch
from sklearn import svm,metrics#svm用于调用分类器，metrics用于生成误差矩阵
from sklearn.model_selection import train_test_split
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

transform = T.Compose([      
            T.Resize([64,64])
    ])
dataset = ImageFolder('Train', transform=transform)
#利用pytorch中的imagefolder函数读取分文件夹保存的图片数据，并将图片大小统一为64x64像素。并利用opencv中的HOGDescriptor函数进行图片的特征提取。
#注意到这里的Train文件与本程序文件在同一根目录，助教运行时可能要修改读取路径
winSize = (16,16)
blockSize = (8,8)
blockStride = (4,4)
cellSize = (4,4)
nbins = 6
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)

winStride=(8,8)
padding = (0,0)
#初始化HOGdescriptor参数，winSize为16x16，即将整张图片分割成16x16的窗口，窗口扫描步长为winStride=8，并在每个窗口中，用大小为8x8，每次移动步长为4的block进行扫描，而每个block又分为四个4x4的cell，在每个cell中计算6个方向（即每个方向30°）的梯度大小。
x=numpy.zeros((14463,10584))
y=numpy.zeros(14463)
#最终得到：窗口数量x每个窗口block个数x每个block中cell个数x梯度方向个数即49x9x4x6=10584维的向量。
for i in range(14463):
    t=numpy.array(dataset[i][0])
    a=hog.compute(t,winStride,padding)
    a=a.flatten()
    x[i]=a
    y[i]=dataset[i][1]
print(x.shape)
#对数据集进行处理，得到14463x10584大小的数据x与14463x1大小的标签y
data=(x,y)
classifier = svm.SVC()
# Split data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True,stratify=y)
#利用scikitlearn中的svm构建分类器，考虑到实验给出的训练与数据集之比大约为4:1，利用train_test_split函数将数据按4：1分为训练集与测试集两个部分。其中shuffle表示对数据进行随机打乱，stratify=y表示按照y标签中类别的比例进行划分。

classifier.fit(X_train,y_train)
predicted = classifier.predict(X_test)
#  模型训练与预测。
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(y_test, predicted)))
disp = metrics.plot_confusion_matrix(classifier, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)
plt.show()
#  给出预测结果与误差矩阵
from sklearn.externals import joblib
joblib.dump(classifier,'svm.model')
#保存模型。