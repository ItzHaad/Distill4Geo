# %%
import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from copy import deepcopy

import matplotlib.pyplot as plt
from PIL import Image
import random
from random import randint
import pandas as pd
import timm
from torch.cuda.amp import autocast
from collections import defaultdict
import copy

from fastervit import create_model
from lightning import LightningModule, Trainer, LightningDataModule
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from distill_transforms import get_transforms_train, get_transforms_val

from lightning.pytorch.loggers import CSVLogger

# %%
train_df = pd.read_csv('CVUSA/train.csv')
train_image_pairs_details = train_df.values.tolist()

val_df = pd.read_csv('CVUSA/val.csv')
val_image_pairs_details = val_df.values.tolist()
# %%
prob_rotate: float = 0.75          # rotates the sat image and panorama images simultaneously
prob_flip: float = 0.5             # flipping the sat image and panorama images simultaneously
teacher_sat_size: tuple = (384, 384)
teacher_street_size: tuple = (384, 768)
student_sat_size: tuple = (224, 224)
student_street_size: tuple = (224, 224)

sat_transforms_train, street_transforms_train = get_transforms_train(teacher_sat_size, teacher_street_size, student_sat_size, student_street_size)
sat_transforms_val_teacher, street_transforms_val_teacher = get_transforms_val(teacher_sat_size, teacher_street_size)
sat_transforms_val_student, street_transforms_val_student = get_transforms_val(student_sat_size, student_street_size)

class VigorDataset(Dataset):
    def __init__(self, split = 'train', same_area = True):
        self.split =  split
        if same_area:
            self.cities = ['Chicago', 'NewYork', 'SanFrancisco', 'Seattle']
        else:
            if split == 'train':
                self.cities = ['NewYork', 'Seattle'] 
            else:
                self.cities = ['Chicago', 'SanFrancisco']

        print(self.cities)
        sat_list = []
        for city in self.cities:
            df_tmp = pd.read_csv(f'VIGOR/splits/{city}/satellite_list.txt', header=None,  sep='\s+')
            df_tmp = df_tmp.rename(columns={0: "sat"})
            df_tmp["path"] = df_tmp.apply(lambda x: f'VIGOR/{city}/satellite/{x.sat}', axis=1)
            sat_list.append(df_tmp)
        self.df_sat = pd.concat(sat_list, axis=0).reset_index(drop=True)
        
        # idx for complete train and test independent of mode = train or test
        sat2idx = dict(zip(self.df_sat.sat, self.df_sat.index))
        self.idx2sat = dict(zip(self.df_sat.index, self.df_sat.sat))
        self.idx2sat_path = dict(zip(self.df_sat.index, self.df_sat.path))
        
        
        # panorama dependent on mode 'train' or 'test'
        panorama_list = []
        for city in self.cities:

            if same_area:
                df_tmp = pd.read_csv(f'VIGOR/splits/{city}/same_area_balanced_{split}.txt', header=None,  sep='\s+')
            else:
                df_tmp = pd.read_csv(f'VIGOR/splits/{city}/pano_label_balanced.txt', header=None,  sep='\s+')
            
            df_tmp = df_tmp.loc[:, [0, 1, 4, 7, 10]].rename(columns={0:  "panorama",
                                                                     1:  "sat",
                                                                     4:  "sat_np1",
                                                                     7:  "sat_np2",
                                                                     10: "sat_np3"})
            
            df_tmp["path_panorama"] = df_tmp.apply(lambda x: f'VIGOR/{city}/panorama/{x.panorama}', axis=1)
            df_tmp["path_sat"] = df_tmp.apply(lambda x: f'VIGOR/{city}/satellite/{x.sat}', axis=1)
            
            for sat_n in ["sat", "sat_np1", "sat_np2", "sat_np3"]:
                df_tmp[f'{sat_n}'] = df_tmp[f'{sat_n}'].map(sat2idx)
                
            panorama_list.append(df_tmp) 
        self.df_panorama = pd.concat(panorama_list, axis=0).reset_index(drop=True)
        
        # idx for split train or test dependent on mode = train or test
        self.idx2panorama = dict(zip(self.df_panorama.index, self.df_panorama.panorama))
        self.idx2panorama_path = dict(zip(self.df_panorama.index, self.df_panorama.path_panorama))
                
      
        self.pairs = list(zip(self.df_panorama.index, self.df_panorama.sat))
        self.idx2pairs = defaultdict(list)
        
        # for a unique sat_id we can have 1 or 2 panorama views as gt
        for pair in self.pairs:      
            self.idx2pairs[pair[1]].append(pair)
            
            
        self.label = self.df_panorama[["sat", "sat_np1", "sat_np2", "sat_np3"]].values 
        
        self.samples = copy.deepcopy(self.pairs)
    
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        idx_panorama, idx_sat = self.samples[idx]
        sat_image_path = self.idx2sat_path[idx_sat]
        street_image_path = self.idx2panorama_path[idx_panorama]
        is_flip = False
        if np.random.random() < prob_flip:
            is_flip = True

        # Process the satellite image
        with Image.open(sat_image_path).convert('RGB') as sat_img:
            if is_flip and self.split == 'train':
                sat_img = sat_img.transpose(Image.FLIP_LEFT_RIGHT)
            sat_img = np.asarray(sat_img)
            if self.split == 'train':
                sat_img_teacher, sat_img_student = sat_transforms_train(sat_img)
            else:
                sat_img_teacher, sat_img_student = sat_transforms_val_teacher(image = sat_img)['image'], sat_transforms_val_student(image = sat_img)['image'] 
            
        # Process the street image
        with Image.open(street_image_path).convert('RGB') as street_img:
            if is_flip and self.split == 'train':
                street_img = street_img.transpose(Image.FLIP_LEFT_RIGHT)
            street_img = np.asarray(street_img)
            if self.split == 'train':
                street_img_teacher, street_img_student = street_transforms_train(street_img)
            else:
                street_img_teacher, street_img_student = street_transforms_val_teacher(image = street_img)['image'], street_transforms_val_student(image = street_img)['image']

        if np.random.random() < prob_rotate and self.split == 'train':
        
            r = np.random.choice([1,2,3])
            
            # rotate sat img 90 or 180 or 270
            sat_img_teacher = torch.rot90(sat_img_teacher, k=r, dims=(1, 2))
            sat_img_student = torch.rot90(sat_img_student, k=r, dims=(1, 2))
            
            # use roll for panorama view if rotate sat view
            c, h, w = street_img_teacher.shape
            shifts = - w//4 * r
            street_img_teacher = torch.roll(street_img_teacher, shifts=shifts, dims=2)
            
            c, h, w = street_img_student.shape
            shifts = - w//4 * r
            street_img_student = torch.roll(street_img_student, shifts=shifts, dims=2) 
            
        return sat_img_teacher, street_img_teacher, sat_img_student, street_img_student
    
train_dataset = VigorDataset(split = "train", same_area= False)
val_dataset = VigorDataset(split = "test", same_area= False)

train_loader = DataLoader(train_dataset, batch_size= 36, num_workers = 8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size = 36, num_workers = 8, shuffle=False)

# %%
class TimmModel(nn.Module):

    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=383):
                 
        super(TimmModel, self).__init__()
        
        self.img_size = img_size
        
        if "vit" in model_name:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size) 
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        
    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config
    
    
    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

        
    def forward(self, img1, img2=None):
        
        if img2 is not None:
       
            image_features1 = self.model(img1)     
            image_features2 = self.model(img2)
            
            return image_features1, image_features2            
              
        else:
            image_features = self.model(img1)
             
            return image_features

# %%
class ContrastiveModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.strict_loading = False
        self.teacher_model = TimmModel('convnext_base.fb_in22k_ft_in1k_384', pretrained=False, img_size= 384)
        self.teacher_model.load_state_dict(torch.load("pretrained/vigor_same/convnext_base.fb_in22k_ft_in1k_384/weights_e40_0.7786.pth", weights_only=True))
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        self.sat_model = create_model('faster_vit_0_224', pretrained=True, model_path="/tmp/faster_vit_0.pth.tar")
        self.sat_linear = nn.Linear(1000, 1024)
        
        self.street_model = create_model('faster_vit_0_224', pretrained=True, model_path="/tmp/faster_vit_0.pth.tar")
        self.street_linear = nn.Linear(1000, 1024)
        
        self.loss = nn.CosineEmbeddingLoss()
                
    def forward(self, teacher_sat_feat, teacher_street_feat, stud_sat_feat, stud_street_feat):
        student_sat_features = self.sat_linear(self.sat_model(stud_sat_feat))
        student_street_features = self.street_linear(self.street_model(stud_street_feat))
        
        with torch.no_grad():
            with autocast():
                teacher_sat_features = self.teacher_model(teacher_sat_feat)
                teacher_street_features = self.teacher_model(teacher_street_feat)
            
        
        loss_sat = self.loss(student_sat_features, teacher_sat_features, torch.ones(student_sat_features.shape[0], device= self.device))
        loss_street = self.loss(student_street_features, teacher_street_features, torch.ones(student_street_features.shape[0], device= self.device))
        
        return (loss_sat + loss_street) / 2

    
    def training_step(self, batch, batch_idx):
        teacher_sat_feat, teacher_street_feat, stud_sat_feat, stud_street_feat = batch
        loss = self(teacher_sat_feat, teacher_street_feat, stud_sat_feat, stud_street_feat)
    
        #n = logits.size(0)
	
        # -1 for off-diagonals and 1 for diagonals
        #labels = 2 * torch.eye(n, device=logits.device) - 1
        
        # pairwise sigmoid loss
        #loss= -torch.sum(F.logsigmoid(labels * logits)) / n

        # alpha = self.current_epoch / self.trainer.max_epochs
        
        # loss = (1 - alpha) * mse_loss + alpha * c_loss
        
        self.log("train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        teacher_sat_feat, teacher_street_feat, stud_sat_feat, stud_street_feat = batch
        loss = self(teacher_sat_feat, teacher_street_feat, stud_sat_feat, stud_street_feat)

        #n = logits.size(0)
	
        # # -1 for off-diagonals and 1 for diagonals
        #labels = 2 * torch.eye(n, device=logits.device) - 1
        
        # # pairwise sigmoid loss
        #loss= -torch.sum(F.logsigmoid(labels * logits)) / n

        self.log("val_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
            
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr= 1e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs = 1, max_epochs = 512)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}] #optimizer

    def on_save_checkpoint(self, checkpoint):
        # Exclude teacher model's weights from checkpoint
        checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items() if 'teacher_model' not in k}
                
#%%

seed_everything(42, workers=True)
checkpoint_callback = ModelCheckpoint(filename= 'tinyvit', monitor="val_loss")
rich_callback = RichProgressBar(leave = True)
logger = CSVLogger("logs", name="KD_EMBEDDING_CD_ONLY_VIGOR_CROSS")

model = ContrastiveModel()

trainer = Trainer(
    max_epochs=512,
    devices=4,
    accelerator="gpu", 
    strategy="ddp",
    callbacks=[checkpoint_callback, rich_callback],
    precision="32",
    deterministic=True,
    accumulate_grad_batches=10,
    gradient_clip_val=1.0,
    logger=logger,
    )
trainer.fit(model, train_loader, val_loader, ckpt_path = "logs/KD_EMBEDDING_CD_ONLY_VIGOR_CROSS/version_7/checkpoints/tinyvit.ckpt")
# %%