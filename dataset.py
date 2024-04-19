import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import random
from random import randrange
import cv2
import numpy as np

from itertools import cycle
import os

class Get_UDataset(Dataset):
    def __init__(self, train_dir_f,train_dir_b,train_name_list, is_patch=True, is_tno=False, transform=None):
        super(Get_UDataset, self).__init__()
        
        self.train_name_list = train_name_list
        self.train_dir_f = train_dir_f
        self.train_dir_b = train_dir_b
        self.transform = transform
        self.is_patch = is_patch
        self.is_tno = is_tno

    def __getitem__(self, index):
        train_name = self.train_name_list[index]
        f = cv2.imread(self.train_dir_f + train_name, cv2.IMREAD_GRAYSCALE)
        f = np.expand_dims(f,axis=-1)
        #f = cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb)
        #f = f[:, :, 0:1]
        if self.is_tno:
            b = cv2.imread(self.train_dir_b + train_name, cv2.IMREAD_GRAYSCALE)
            b = np.expand_dims(b,axis=-1)
        else:
            b = cv2.imread(self.train_dir_b + train_name)
            b = cv2.cvtColor(b, cv2.COLOR_BGR2YCrCb)
            b = b[:, :, 0:1]
        if self.is_patch:
            f, b = self.get_patch(f, b, patch_size=64)
        # ------------------To tensor------------------#
        if self.transform is not None:
            tran = transforms.ToTensor()
            f = tran(f)
            b = tran(b)
            return f,b

    def __len__(self):
        return len(self.train_name_list)
    
    def get_patch(self, img_in, img_in1, patch_size):
        h, w = img_in.shape[:2]
        stride = patch_size
        x = random.randint(0, w - stride)
        y = random.randint(0, h - stride)
        img_in = img_in[y:y + stride, x:x + stride, :]
        img_in1 = img_in1[y:y + stride, x:x + stride, :]
        return img_in,img_in1

class Get_UDataset_RGB(Dataset):
    def __init__(self, train_dir_f,train_dir_b,train_name_list, is_patch=True, transform=None):
        super(Get_UDataset_RGB, self).__init__()
        
        self.train_name_list = train_name_list
        self.train_dir_f = train_dir_f
        self.train_dir_b = train_dir_b
        self.transform = transform
        self.is_patch = is_patch

    def __getitem__(self, index):
        train_name = self.train_name_list[index]
        f = cv2.imread(self.train_dir_f + train_name)
        b = cv2.imread(self.train_dir_b + train_name)
        if self.is_patch:
            f, b = self.get_patch(f, b, patch_size=64)
        # ------------------To tensor------------------#
        if self.transform is not None:
            tran = transforms.ToTensor()
            f = tran(f)
            b = tran(b)
            return f,b

    def __len__(self):
        return len(self.train_name_list)
    
    def get_patch(self, img_in, img_in1, patch_size):
        h, w = img_in.shape[:2]
        stride = patch_size
        x = random.randint(0, w - stride)
        y = random.randint(0, h - stride)
        img_in = img_in[y:y + stride, x:x + stride, :]
        img_in1 = img_in1[y:y + stride, x:x + stride, :]
        return img_in,img_in1

class Get_SDataset_RGB(Dataset):
    def __init__(self, train_dir_f,train_dir_b,train_dir_g,train_name_list, is_patch=True, transform=None):
        super(Get_SDataset_RGB, self).__init__()
        
        self.train_name_list = train_name_list
        self.train_dir_f = train_dir_f
        self.train_dir_b = train_dir_b
        self.train_dir_g = train_dir_g
        self.transform = transform
        self.is_patch = is_patch

    def __getitem__(self, index):

        train_name = self.train_name_list[index]
        f = cv2.imread(self.train_dir_f + train_name)
        if "A.png" in train_name:
            b = cv2.imread(self.train_dir_b + train_name.replace("A","B"))
            clear = cv2.imread(self.train_dir_g + train_name.replace("A","F"))
        else:
            b = cv2.imread(self.train_dir_b + train_name)
            clear = cv2.imread(self.train_dir_g + train_name)
        if self.is_patch:
            f, b, clear = self.get_patch(f, b,clear, patch_size=128)
        # ------------------To tensor------------------#
        if self.transform is not None:
            tran = transforms.ToTensor()
            f = tran(f)
            b = tran(b)
            clear = tran(clear) 
            return f,b,clear
    def __len__(self):
        return len(self.train_name_list)
    
    def get_patch(self, img_in, img_in1,img_tar, patch_size):
        h, w = img_in.shape[:2]

        stride = patch_size

        x = random.randint(0, w - stride)
        y = random.randint(0, h - stride)

        img_in = img_in[y:y + stride, x:x + stride, :]
        img_in1 = img_in1[y:y + stride, x:x + stride, :]
        img_tar = img_tar[y:y + stride, x:x + stride, :]

        return img_in,img_in1, img_tar


class Get_SDataset(Dataset):
    def __init__(self, train_dir_f,train_dir_b,train_dir_g,train_name_list, is_patch=True, transform=None):
        super(Get_SDataset, self).__init__()
        
        self.train_name_list = train_name_list
        self.train_dir_f = train_dir_f
        self.train_dir_b = train_dir_b
        self.train_dir_g = train_dir_g
        self.transform = transform
        self.is_patch = is_patch

    def __getitem__(self, index):

        train_name = self.train_name_list[index]
        f = cv2.imread(self.train_dir_f + train_name)
        f = cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb)
        f = f[:, :, 0:1]
        if "A.png" in train_name:
            b = cv2.imread(self.train_dir_b + train_name.replace("A","B"))
            b = cv2.cvtColor(b, cv2.COLOR_BGR2YCrCb)
            b = b[:, :, 0:1]
            clear = cv2.imread(self.train_dir_g + train_name.replace("A","F"))
            clear = cv2.cvtColor(clear, cv2.COLOR_BGR2YCrCb)
            clear = clear[:, :, 0:1]
        else:
            b = cv2.imread(self.train_dir_b + train_name)
            b = cv2.cvtColor(b, cv2.COLOR_BGR2YCrCb)
            b = b[:, :, 0:1]
            clear = cv2.imread(self.train_dir_g + train_name)
            clear = cv2.cvtColor(clear, cv2.COLOR_BGR2YCrCb)
            clear = clear[:, :, 0:1]
        if self.is_patch:
            f, b, clear = self.get_patch(f, b,clear, patch_size=128)
        # ------------------To tensor------------------#
        if self.transform is not None:
            tran = transforms.ToTensor()
            f = tran(f)
            b = tran(b)
            clear = tran(clear) 
            return f,b,clear
    def __len__(self):
        return len(self.train_name_list)
    
    def get_patch(self, img_in, img_in1,img_tar, patch_size):
        h, w = img_in.shape[:2]

        stride = patch_size

        x = random.randint(0, w - stride)
        y = random.randint(0, h - stride)

        img_in = img_in[y:y + stride, x:x + stride, :]
        img_in1 = img_in1[y:y + stride, x:x + stride, :]
        img_tar = img_tar[y:y + stride, x:x + stride, :]

        return img_in,img_in1, img_tar

class Get_MEF_Dataset(Dataset):
    def __init__(self, train_folder_source,train_dir_gt, is_patch=True, transform=None, patch_size=128):
        super(Get_MEF_Dataset, self).__init__()
        
        self.train_folder_source = train_folder_source
        self.train_dir_gt = train_dir_gt
        self.transform = transform
        self.is_patch = is_patch
        self.source_dir = os.listdir(train_folder_source)
        self.source_dir.sort()
        self.gt_name_list = os.listdir(train_dir_gt)
        self.gt_name_list.sort()
        self.patch_size = patch_size

    def __getitem__(self, index):

        image_names = os.listdir(self.train_folder_source+self.source_dir[index])
        image_index = random.randint(0,(len(image_names)-1)//2)
        under = cv2.imread(self.train_folder_source+self.source_dir[index]+'/'+image_names[image_index])
        over = cv2.imread(self.train_folder_source+self.source_dir[index]+'/'+image_names[len(image_names)-image_index-1])
        gt = cv2.imread(self.train_dir_gt+self.gt_name_list[index])
        #print(index)
        #print(gt.shape,under.shape,over.shape)
        under = cv2.cvtColor(under, cv2.COLOR_BGR2YCrCb)
        #under = under[:, :, 0:1]
        over = cv2.cvtColor(over, cv2.COLOR_BGR2YCrCb)
        #over = over[:, :, 0:1]
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2YCrCb)
        #gt = gt[:, :, 0:1]
        if self.is_patch:
            under, over, gt = self.get_patch(under, over,gt, patch_size=self.patch_size)
        # ------------------To tensor------------------#
        if self.transform is not None:
            tran = transforms.ToTensor()
            under = tran(under)
            over = tran(over)
            gt = tran(gt) 
            return under,over,gt
    def __len__(self):
        return len(self.gt_name_list)
    
    def get_patch(self, img_in, img_in1,img_tar, patch_size):
        h, w = img_in.shape[:2]

        stride = patch_size

        x = random.randint(0, w - stride)
        y = random.randint(0, h - stride)

        img_in = img_in[y:y + stride, x:x + stride, :]
        img_in1 = img_in1[y:y + stride, x:x + stride, :]
        img_tar = img_tar[y:y + stride, x:x + stride, :]

        return img_in,img_in1, img_tar

class Get_MEF_Dataset_RGB(Dataset):
    def __init__(self, train_folder_source,train_dir_gt, is_patch=True, transform=None):
        super(Get_MEF_Dataset_RGB, self).__init__()
        
        self.train_folder_source = train_folder_source
        self.train_dir_gt = train_dir_gt
        self.transform = transform
        self.is_patch = is_patch
        self.source_dir = os.listdir(train_folder_source)
        self.source_dir.sort()
        self.gt_name_list = os.listdir(train_dir_gt)
        self.gt_name_list.sort()

    def __getitem__(self, index):

        image_names = os.listdir(self.train_folder_source+self.source_dir[index])
        image_index = random.randint(0,(len(image_names)-1)//2)
        under = cv2.imread(self.train_folder_source+self.source_dir[index]+'/'+image_names[image_index])
        over = cv2.imread(self.train_folder_source+self.source_dir[index]+'/'+image_names[len(image_names)-image_index-1])
        gt = cv2.imread(self.train_dir_gt+self.gt_name_list[index])
        if self.is_patch:
            under, over, gt = self.get_patch(under, over,gt, patch_size=128)
        # ------------------To tensor------------------#
        if self.transform is not None:
            tran = transforms.ToTensor()
            under = tran(under)
            over = tran(over)
            gt = tran(gt) 
            return under,over,gt
    def __len__(self):
        return len(self.gt_name_list)
    
    def get_patch(self, img_in, img_in1,img_tar, patch_size):
        h, w = img_in.shape[:2]

        stride = patch_size

        x = random.randint(0, w - stride)
        y = random.randint(0, h - stride)

        img_in = img_in[y:y + stride, x:x + stride, :]
        img_in1 = img_in1[y:y + stride, x:x + stride, :]
        img_tar = img_tar[y:y + stride, x:x + stride, :]

        return img_in,img_in1, img_tar