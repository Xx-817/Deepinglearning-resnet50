# Deepinglearning-resnet50  
基于深度学习resnet50进行垃圾分类  

**本次用于训练的数据集由于学校要求暂不公开**  
 
```
#所用的函数 
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms

import torch.optim as optim
import torchvision.models as models

import PIL.Image as Image
#将图片进行归一化处理
image_size = (224,224)
data_transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
train_data=dset.ImageFolder(root="data/train",transform=data_transform)
# 数据集长度
totallen = len(train_data)
print('train data length:',totallen)
trainlen = int(totallen*0.7)
vallen = totallen - trainlen
train_db,val_db=torch.utils.data.random_split(train_data,[trainlen,vallen])
print('train:',len(train_db),'validation:',len(val_db))
# batch size
bs=32
# 训练集
train_loader=torch.utils.data.DataLoader(train_db,batch_size=bs, shuffle=True,num_workers=2)
# 验证集
val_loader=torch.utils.data.DataLoader(val_db,batch_size=bs, shuffle=True,num_workers=2)
def get_num_correct(out, labels):
    return out.argmax(dim=1).eq(labels).sum().item()
batch = next(iter(train_loader))
resnet50 = models.resnet50(pretrained=True)
model = resnet50
n_classes = len(train_data.classes)
model.fc = nn.Linear(2048, n_classes)
import torch.nn.init as init

for name, module in model._modules.items():
    if(name=='fc'):
        # print(module.weight.shape)
        init.kaiming_uniform_(module.weight, a=0, mode='fan_in')
device=torch.device("cpu")
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
epoch_num = 5
model = model.to(device)
for epoch in range(epoch_num):
    total_loss=0
    total_correct=0
    val_correct=0
    for batch in train_loader:#GetBatch
        images,labels=batch
        images = images.to(device)
        labels = labels.to(device)
        outs=model(images)#PassBatch
        loss=F.cross_entropy(outs,labels)#CalculateLoss
        optimizer.zero_grad()
        loss.backward()#CalculateGradients
        optimizer.step()#UpdateWeights
        total_loss+=loss.item()
        total_correct+=get_num_correct(outs,labels)
    for batch in val_loader:
        images,labels=batch
        images = images.to(device)
        labels = labels.to(device)
        outs=model(images)
        val_correct+=get_num_correct(outs,labels)
    print("loss:",total_loss,"train_correct:",total_correct/trainlen, "val_correct:",val_correct/vallen)
torch.save(model, 'trash.pkl')
#模型应用于测试集
model = torch.load('trash.pkl')
model.eval()
filename = 'data/test/101.jpg'
input_image = Image.open(filename).convert('RGB')
input_tensor = data_transform(input_image)
input_batch = input_tensor.unsqueeze(0) 
model.to('cpu')
with torch.no_grad():
    output = model(input_batch)
prob = F.softmax(output[0], dim=0)
indexs = torch.argsort(-prob)
topk = 1
a=['hazardous_waste_dry_battery',
 'hazardous_waste_expired_drugs',
 'hazardous_waste_ointment',
 'kitchen_waste_bone',
 'kitchen_waste_eggshell',
 'kitchen_waste_fish_bone',
 'kitchen_waste_fruit_peel',
 'kitchen_waste_meal',
 'kitchen_waste_pulp',
 'kitchen_waste_tea',
 'kitchen_waste_vegetable',
 'other_garbage_bamboo_chopsticks',
 'other_garbage_cigarette',
 'other_garbage_fast_food_box',
 'other_garbage_flowerpot',
 'other_garbage_soiled_plastic',
 'other_garbage_toothpick',
 'recyclables_anvil',
 'recyclables_bag',
 'recyclables_bottle',
 'recyclables_can',
 'recyclables_cardboard',
 'recyclables_cosmetic_bottles',
 'recyclables_drink_bottle',
 'recyclables_edible_oil_barrel',
 'recyclables_glass_cup',
 'recyclables_metal_food_cans',
 'recyclables_old_clothes',
 'recyclables_paper_bags',
 'recyclables_pillow',
 'recyclables_plastic_bowl',
 'recyclables_plastic_hanger',
 'recyclables_plug_wire',
 'recyclables_plush_toys',
 'recyclables_pot',
 'recyclables_powerbank',
 'recyclables_seasoning_bottle',
 'recyclables_shampoo_bottle',
 'recyclables_shoes',
 'recyclables_toys']
for i in range(topk):
    print("label:", a[indexs[i].item()], " prob: ", prob[indexs[i]])
 ```
 训练结果和测试结果如图
![训练过程](ttps://github.com/Xx-817/Deepinglearning-resnet50/master/1.png)
