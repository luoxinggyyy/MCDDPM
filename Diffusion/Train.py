
import os
from typing import Dict
import pyvista as pv
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Diffusion.Model1 import UNet
from Scheduler import GradualWarmupScheduler
from torch.cuda.amp import GradScaler as GradScaler
from torch.cuda.amp import autocast
class MyDataset(Dataset): #继承Dataset
    def __init__(self, path_dir, transform=None): #初始化一些属性
        self.path_dir = path_dir #文件路径
        self.transform = transform #对图形进行处理，如标准化、截取、转换等
        self.images = os.listdir(self.path_dir)#把路径下的所有文件放在一个列表中

    def __len__(self):#返回整个数据集的大小
        return len(self.images)

    def __getitem__(self,index):#根据索引index返回图像及标签
        image_index = self.images[index]#根据索引获取图像文件名称
        img_path = os.path.join(self.path_dir, image_index)#获取图像的路径或目录
        img = Image.open(img_path).convert('RGB')# 读取图像

        # 根据目录名称获取图像标签（cat或dog）
        label = img_path.split('\\')[-1]
        label = label.strip('png')[:-1]
        # label = label.split('.')[:-1]
        label = np.float32(label)
        #把字符转换为数字cat-0，dog-1
        # label = 1 if 'dog' in label else 0
        if self.transform is not None:
            img = self.transform(img)
        return img,label

class MyDataset3d(Dataset): #继承Dataset
    def __init__(self, path_dir): #初始化一些属性
        self.path_dir = path_dir #文件路径
        self.images = os.listdir(self.path_dir)#把路径下的所有文件放在一个列表中

    def __len__(self):#返回整个数据集的大小
        return len(self.images)

    def __getitem__(self,index):#根据索引index返回图像及标签
        image_index = self.images[index]#根据索引获取图像文件名称
        img_path = os.path.join(self.path_dir, image_index)#获取图像的路径或目录
        img = np.load(img_path)# 读取图像
        img   = np.array(img, dtype=np.float32)
        label = img_path.split('\\')[-1]
        label = label.strip('npy')[:-1]
        # label = label.split('.')[:-1]
        label = np.float32(label)
        return img,label


def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # dataset
    tf = transforms.Compose(
    [
        # transforms.Resize(64),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        
    ])
    dataset = MyDataset('./data3',transform=tf)
    dataset1 = MyDataset3d('F:\\deep-learning-code\\image-generation-code\\video-diffusion-pytorch-main\\data')
    dataloader = DataLoader(
        dataset1, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    # model setup
    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    scaler = GradScaler()
    # start training
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                # train
                optimizer.zero_grad()
                x_0 = images.to(device)
                x_0 = torch.unsqueeze(x_0, dim=1)
                # loss = trainer(x_0).sum() / 1000.
                # loss.backward()
                # torch.nn.utils.clip_grad_norm_(
                #     net_model.parameters(), modelConfig["grad_clip"])
                # optimizer.step()
                # tqdmDataLoader.set_postfix(ordered_dict={
                #     "epoch": e,
                #     "loss: ": loss.item(),
                #     "img shape: ": x_0.shape,
                #     "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                # })
                with autocast():
                    loss = trainer(x_0).sum() / 1000.
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        torch.save(net_model.state_dict(), os.path.join(
            modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))
def save_3d(a,now,patha,pathb):
    b = torch.squeeze(a, dim=1).cpu()
    columns = now//2
    rows = 2
    t = np.arange(0,rows*columns).reshape(rows,columns)
    p = pv.Plotter(shape=(rows, columns))
    path = patha+pathb
    for i in range(rows):
        for j in range(columns):
            grid = pv.UniformGrid(dims=(65,65,65))
            grid.cell_data["values"] = b[t[i,j],:,:,:].flatten() 
            clipped = grid.clip_box(factor=0.5)
            p.subplot(i,j)
            p.add_mesh(clipped, cmap='gray', show_scalar_bar=False)

    #         p.camera_position = [-5, 5, 5]
            p.window_size = [1000,500]
            #p.camera.zoom(1.1)
    # p.save_graphic('../figures/training_samples.eps',raster=False )
    # p.save_graphic('../figures/training_samples.eps') 
    p.show() 
    p.screenshot('rock3d.eps', transparent_background=True)
    return p
def eval(modelConfig: Dict):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[modelConfig["batch_size"], 1, 64, 64, 64], device=device)
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        # save_image(saveNoisy, os.path.join(
        #     modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
        # save_3d(saveNoisy, modelConfig["nrow"], modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"])
        sampledImgs = sampler(noisyImage)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        # sampledImgs = torch.where(sampledImgs<torch.mean(sampledImgs), 0, 255)
        # save_image(sampledImgs, os.path.join(
        #     modelConfig["sampled_dir"],  modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])
        save_3d(sampledImgs, modelConfig["nrow"], modelConfig["sampled_dir"], modelConfig["sampledImgName"])