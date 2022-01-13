## Deepinglearning-resnet50
基于深度学习resnet50进行垃圾分类  
**trash.pkl为本地训练好的模型**  
data数据集包含测试集和训练集，用者自提  
rubbish.py是基于streamlit制作的demo  
rubbish.py  
```
import streamlit as st
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms

import torch.optim as optim
import torchvision.models as models
import PIL.Image as Image
st.title("垃圾分类")

image_size = (224,224)
data_transform=transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
train_data=dset.ImageFolder(root="data/train",transform=data_transform)

def get_num_correct(out, labels):
    return out.argmax(dim=1).eq(labels).sum().item()
st.markdown("导入模型")
with st.echo():
    model = torch.load('trash.pkl',map_location='cpu')
    model.eval()
import pandas as pd
import os
import csv
st.markdown("导入测试集并查看测试集大小")
# 将id读取
path_test = 'data/test/'
# 训练出来垃圾类别
dirs = os.listdir(path_test)
st.write(len(dirs))
#处理测试数据
import cv2
def testrunning():
    result=[]
    #filename=path_test+str(0)+'.jpg'Q
    filename= st.file_uploader("请上传你要识别的图片：", type="jpg")
    if filename is None:
        filename=path_test+str(0)+'.jpg'
    st.image(filename)
    input_image = Image.open(filename).convert('RGB')
    input_tensor = data_transform(input_image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(input_batch)
        prob = F.softmax(output[0], dim=0)
        indexs = torch.argsort(-prob)
        topk = 1
        for j in range(topk):
            result.append(train_data.classes[indexs[j].item()])
    return result
st.write(testrunning())
```
demo如图
![demo]()
