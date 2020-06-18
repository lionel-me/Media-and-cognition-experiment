import json
from PIL import Image
import os
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import transforms as T

file_path = 'train_annotations.json'#file_path可能需要修改
with open(file_path) as f:
    js = json.load(f)  # js是转换后的字典

class traffic_sign_Dataset(object):#数据集类
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "train"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "train", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        temp=imgs[idx][:-4]
        num_objs = len(js['imgs'][temp]['objects'])
        boxes = []
        for i in range(num_objs):#提取json文件中的bbox信息
            xmin=js['imgs'][temp]['objects'][i]['bbox']['xmin']
            xmax=js['imgs'][temp]['objects'][i]['bbox']['xmax']
            ymin=js['imgs'][temp]['objects'][i]['bbox']['ymin']
            ymax=js['imgs'][temp]['objects'][i]['bbox']['ymax']
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = torch.tensor([int(imgs[idx][:-4])])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def get_instance_segmentation_model(num_classes):#模型函数，载入预训练模型
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2  # 1 class (traffic sign) + background

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def get_transform(train):#对图片进行处理，训练集图片进行随机转向，增加模型的鲁棒性
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

dataset = traffic_sign_Dataset('/home/ass02/Datasets/image_exp/Detection/')#所有训练集图片

dataset = traffic_sign_Dataset('/home/ass02/Datasets/image_exp/Detection/', get_transform(train=True))#训练集图片，随机翻转
dataset_test = traffic_sign_Dataset('/home/ass02/Datasets/image_exp/Detection/', get_transform(train=False))#测试集

torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])#将后五十张图片归入测试集，预测模型结果

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)
 
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)
 
device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 2#交通标志与背景两类
 

model = get_instance_segmentation_model(num_classes)#得到预训练过的模型

model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
 
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)#每训练三轮，学习率下降10x
                                            
num_epochs = 10#训练十轮
for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=200)
 
    # update the learning rate
    lr_scheduler.step()
 
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

torch.save(model, 'task4_fasterrcnn.pth')#保存模型参数

