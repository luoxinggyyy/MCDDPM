import pyvista as pv
import os
import pandas as pd
from typing import Dict
import numpy as np
from torchvision.utils import save_image, make_grid
import torch
# import pandas as pd
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
from DiffusionFreeGuidence.DiffusionCondition import GaussianDiffusionSampler, GaussianDiffusionTrainer
from DiffusionFreeGuidence.ModelCondition1 import UNet
from Scheduler import GradualWarmupScheduler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler as GradScaler
from torch.cuda.amp import autocast
from demoto import demotoo
import random

def save_3d(a,now,path):
    b = torch.squeeze(a, dim=1).cpu()
    columns = now//3
    rows = 3
    t = np.arange(0,rows*columns).reshape(rows,columns)
    p = pv.Plotter(shape=(rows, columns))
    for i in range(rows):
        for j in range(columns):
            grid = pv.UniformGrid(dims=(65,65,65))
            grid.cell_data["values"] = b[t[i,j],:,:,:].flatten() 
            clipped = grid.clip_box(factor=0.5)
            p.subplot(i,j)
            p.add_mesh(clipped, cmap='gray', show_scalar_bar=False)
            p.window_size=[400,400]
    #         p.camera_position = [-5, 5, 5]
            #p.camera.zoom(1.1)
    # p.save_graphic('../figures/training_samples.eps',raster=False )
    # p.save_graphic(path,raster=False)
    p.show(screenshot=path,window_size=[4000,4000]) 
    # p.save_graphic('training_samples.pdf',raster=False, painter=False)
    # p.screenshot(path, transparent_background=True)
    return p
class MyDataset(Dataset): 
    def __init__(self, path_dir, transform=None): 
        self.path_dir = path_dir 
        self.transform = transform 
        self.images = os.listdir(self.path_dir)

    def __len__(self):#返回整个数据集的大小
        return len(self.images)

    def __getitem__(self,index):
        image_index = self.images[index]
        img_path = os.path.join(self.path_dir, image_index)
        img = Image.open(img_path).convert('RGB')

    
        label = img_path.split('\\')[-1]
        label = label.strip('png')[:-1]
        # label = label.split('.')[:-1]
        label = np.float32(label)
        if self.transform is not None:
            img = self.transform(img)
        return img,label
class MyDataset3d(Dataset): 
    def __init__(self, path_dir): 
        self.path_dir = path_dir 
        self.images = os.listdir(self.path_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):
        image_index = self.images[index]
        img_path = os.path.join(self.path_dir, image_index)
        img = np.load(img_path)
        img   = np.array(img, dtype=np.float32)
        label = img_path.split('/')[-1]
        label = label.strip('npy')[:-1]
        labelb = np.float32(label.split('_')[1])
        labela = np.float32(label.split('_')[0])
        labelc = np.float32([labela,labelb])
        # label = np.float32(label)
        return img,labelc

def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # dataset
    tf = transforms.Compose(
    [
        transforms.Resize(modelConfig["img_size"]),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    dataset1 = MyDataset3d(modelConfig["path"])
    dataloader = DataLoader(
        dataset1, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    # model setup
    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_dir"], modelConfig["training_load_weight"]), map_location=device), strict=False)
        print("Model weight load down.")
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], betas=(0.9, 0.999),weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=modelConfig["multiplier"],
                                             warm_epoch=modelConfig["epoch"] // 60, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    scaler = GradScaler()
    # start training
    epoch_losses = []

    for e in range(modelConfig["epoch"]):
        epoch_loss = 0
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for i,(images,label2) in enumerate(tqdmDataLoader):
                # train
                x_0 = images.to(device).type(torch.float32)
                label2 = label2.to(device)
                labels = torch.sum(x_0==0,dim=(-1,-2,-3)) /(x_0.shape[-1]**3)
                x_0 = torch.unsqueeze(x_0, dim=1)
                # if np.random.rand() < 0.1:
                #     labels = torch.zeros_like(labels).to(device)
                optimizer.zero_grad()
                with autocast():
                    loss = trainer(x_0, labels,label2).sum() /100.
                    scaler.scale(loss).backward()
                    # scaler.unscale_(optimizer)
                    # torch.nn.utils.clip_grad_norm_(net_model.parameters(), modelConfig["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()


                # loss = trainer(x_0, labels).sum() /10.
                # optimizer.zero_grad()
                # loss.backward()
                # torch.nn.utils.clip_grad_norm_(
                #     net_model.parameters(), modelConfig["grad_clip"])
                # optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                }) 
        warmUpScheduler.step()
        if e%30 == 0:
            torch.save(net_model.state_dict(), os.path.join(
                modelConfig["save_dir"], 'real_our_' + str(e) + "_.pt"))
            epoch_loss/= (i+1)
            epoch_losses.append(epoch_loss)
 


def eval(modelConfig: Dict):
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    setup_seed(20)
    device = torch.device(modelConfig["device"])
    # load model and evaluate
    with torch.no_grad():
        # labels = torch.rand(modelConfig["batch_size"]).to(modelConfig["device"])# 0到1的标签
        labels = torch.empty(modelConfig["batch_size"], dtype=torch.float32).uniform_(modelConfig["label"],modelConfig["label"]+0.00).to(modelConfig["device"]) #0.7到0.8的标签
        labela = torch.empty((modelConfig["batch_size"],1), dtype=torch.float32).uniform_(modelConfig["labelA"],modelConfig["labelA"]+0).to(modelConfig["device"])
        labelb = torch.empty((modelConfig["batch_size"],1), dtype=torch.float32).uniform_(modelConfig["labelB"],modelConfig["labelB"]+0).to(modelConfig["device"])
        label2 = torch.cat((labela,labelb),1)
        print("labels: ", labels, label2)
        model = UNet(T=modelConfig["T"],  ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
        ckpt = torch.load(os.path.join(
            modelConfig["save_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        # print("model load weight done.")
        model.eval()
        dir = 'F:/q/npydata/'
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], w=modelConfig["w"]).to(device)
        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[modelConfig["batch_size"], 1, 64,64,64], device=device)
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)

        sampledImgs = sampler(noisyImage, labels,label2)
        sampledImgs1 = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        save_3d(1-sampledImgs1 , modelConfig["nrow"], dir + 'train2_'+str(modelConfig["label"])+".tif")
        # grid = make_grid(sampledImgs*-1 + 1)
        dir = 'F:/q/npydata/'
        for i,j in enumerate(sampledImgs1):
            j = torch.squeeze(j,dim = 1).cpu().numpy()
            np.save(dir+str(i)+str('.npy'),j)

        bb = []
        for i,j in enumerate(sampledImgs1):
            k = torch.mean(j)
            b = np.round((torch.sum(j<k)/(64*64*64)).cpu().numpy(),3)
            real_label = round(1-labels[i].cpu().numpy(),3)
            # save_image(j, os.path.join(
            # modelConfig["sampled_dir"],  str(f'{real_label}_{b}.png')))
            # save_3d(j , modelConfig["nrow"], modelConfig["sampled_dir"], modelConfig["sampledImgName"])
            bb.append(b) 
        real_label1 = np.round(1-labels.cpu().numpy(),3)
        cc = list(zip(real_label1,bb))
        data1 = pd.DataFrame(cc)
        
        data1.to_csv('cc3.csv')
