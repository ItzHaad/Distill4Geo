# %%
import torch
from torch import nn
import torch.nn.functional as F
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

from fastervit import create_model
from lightning import LightningModule, Trainer, LightningDataModule
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from distill_transforms import get_transforms_train, get_transforms_val

from lightning.pytorch.loggers import CSVLogger

# %%
train_df = pd.read_csv('CVACT/train.csv')
train_image_pairs_details = train_df.values.tolist()

val_df = pd.read_csv('CVACT/val.csv')
val_image_pairs_details = val_df.values.tolist()
# %%
prob_rotate: float = 0.75          # rotates the sat image and ground images simultaneously
prob_flip: float = 0.5             # flipping the sat image and ground images simultaneously
teacher_sat_size: tuple = (384, 384)
teacher_street_size: tuple = (140, 768)
student_sat_size: tuple = (224, 224)
student_street_size: tuple = (224, 224)

sat_transforms_train, street_transforms_train = get_transforms_train(teacher_sat_size, teacher_street_size, student_sat_size, student_street_size)
sat_transforms_val_teacher, street_transforms_val_teacher = get_transforms_val(teacher_sat_size, teacher_street_size)
sat_transforms_val_student, street_transforms_val_student = get_transforms_val(student_sat_size, student_street_size)

class SatStreetDataset(Dataset):
    def __init__(self, image_pairs_details, is_train = False):
        self.image_pairs_details = image_pairs_details
        self.is_train = is_train
        
        
    def __len__(self):
        return len(self.image_pairs_details)

    def __getitem__(self, idx):
        sat_image_path = 'CVACT/'+ self.image_pairs_details[idx][0]
        street_image_path = 'CVACT/'+ self.image_pairs_details[idx][1]
        is_flip = False
        if np.random.random() < prob_flip:
            is_flip = True

        # Process the satellite image
        with Image.open(sat_image_path) as sat_img:
            if is_flip and self.is_train:
                sat_img = sat_img.transpose(Image.FLIP_LEFT_RIGHT)
            sat_img = np.asarray(sat_img)
            # sat_img = sat_img.rotate(randint(0, 360)) if self.is_train else sat_img
            if self.is_train:
                sat_img_teacher, sat_img_student = sat_transforms_train(sat_img)
            else:
                sat_img_teacher, sat_img_student = sat_transforms_val_teacher(image = sat_img)['image'], sat_transforms_val_student(image = sat_img)['image'] 
            
        # Process the street image
        with Image.open(street_image_path) as street_img:
            if is_flip and self.is_train:
                street_img = street_img.transpose(Image.FLIP_LEFT_RIGHT)
            street_img = np.asarray(street_img)
            if self.is_train:
                street_img_teacher, street_img_student = street_transforms_train(street_img)
            else:
                street_img_teacher, street_img_student = street_transforms_val_teacher(image = street_img)['image'], street_transforms_val_student(image = street_img)['image']

        if np.random.random() < prob_rotate and self.is_train:
        
            r = np.random.choice([1,2,3])
            
            # rotate sat img 90 or 180 or 270
            sat_img_teacher = torch.rot90(sat_img_teacher, k=r, dims=(1, 2))
            sat_img_student = torch.rot90(sat_img_student, k=r, dims=(1, 2))
            
            # use roll for ground view if rotate sat view
            c, h, w = street_img_teacher.shape
            shifts = - w//4 * r
            street_img_teacher = torch.roll(street_img_teacher, shifts=shifts, dims=2)
            
            c, h, w = street_img_student.shape
            shifts = - w//4 * r
            street_img_student = torch.roll(street_img_student, shifts=shifts, dims=2) 
            
        return sat_img_teacher, street_img_teacher, sat_img_student, street_img_student
    
train_dataset = SatStreetDataset(train_image_pairs_details, is_train= True)
val_dataset = SatStreetDataset(val_image_pairs_details, is_train= False)

train_loader = DataLoader(train_dataset, batch_size= 48, num_workers = 8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size = 48, num_workers = 8, shuffle=False)

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
        self.teacher_model.load_state_dict(torch.load("pretrained/cvact/convnext_base.fb_in22k_ft_in1k_384/weights_e36_90.8149.pth", weights_only=True))
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
# %%

seed_everything(42, workers=True)
checkpoint_callback = ModelCheckpoint(filename= 'tinyvit', monitor="val_loss")
rich_callback = RichProgressBar(leave = True)
logger = CSVLogger("logs", name="KD_EMBEDDING_CD_ONLY_CVACT")

model = ContrastiveModel()

trainer = Trainer(
    max_epochs=512,
    devices=4,
    accelerator="gpu", 
    strategy="ddp",
    callbacks=[checkpoint_callback, rich_callback],
    precision="32",
    deterministic=True,
    gradient_clip_val=1.0,
    # terminate_on_nan = True,
    logger=logger)
trainer.fit(model, train_loader, val_loader, ckpt_path = 'logs/KD_EMBEDDING_CD_ONLY_CVACT/version_0/checkpoints/tinyvit.ckpt')

# %%