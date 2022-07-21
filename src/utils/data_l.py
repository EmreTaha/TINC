import os
import pickle
import numpy as np
import torch
import random
import math
import pandas as pd

from torch.utils import data
from torchvision.io import read_image
from sklearn.model_selection import KFold, train_test_split

class Dataset(data.Dataset):
    def __init__(self, eye_list, labels, transform=None, df=None):
        self.eye_list = eye_list
        self.labels = labels
        self.transform = transform
        self.df = df

    def __len__(self):
        return len(self.eye_list)

    def __getitem__(self, index):

        eyes_paths = self.eye_list[index]
        eyes = read_image(eyes_paths)

        if self.transform:
            eyes = self.transform(eyes)

        conv_label = self.labels[index]

        return eyes, conv_label

class tempContrDataset(Dataset):
    def __init__(self, eye_list, labels, transform=None, volume=False, n_pair=1, binary=360, absol=True, norm_label = False, min_max=(90,540)):
        super(tempContrDataset,self).__init__(eye_list, labels, transform)
        self.volume = volume
        self.n_pair = n_pair

        all_eyes = []
        time_labels = []
        binary_time_labels = []
        pat_ids = []
        for i in self.eye_list:
            # Each path i consists of .../pat_id/visit_day/Bscan.png
            pat_id = i.split('/')[-3]
            visit = i.split('/')[-2]
            b_slice = i.split('/')[-1]

            matching = [s for s in self.eye_list if (pat_id in s) and (b_slice==s.split('/')[-1]) and (abs(int(visit)-int(s.split('/')[-2]))>=min_max[0]) and (abs(int(visit)-int(s.split('/')[-2]))<=min_max[1])]
            j=0
            while len(matching) != 0:
                candidate = random.choice(matching)
                matching.remove(candidate)
                candid_tuple = tuple(sorted((i, candidate)))
                if candid_tuple not in all_eyes and (i, candidate) not in all_eyes: #This could be redundant but just to be sure

                    label = int(visit)-int(candidate.split('/')[-2])
                    label_sign = math.copysign(1,label)
                    norm_time = np.float32((abs(label)-min_max[0])/(min_max[1]-min_max[0])) #min-max normalizer

                    if absol:
                        label = abs(label)
                        all_eyes.append(candid_tuple)  #The order doesnt matter
                    else:
                        norm_time = (label_sign*(norm_time+1e-6) + 1)/2.0 # range is -1 to +1. So we add 1 -> 0-2, divide by 2 -> 0-1
                        all_eyes.append((i, candidate)) #Order matters

                    # Binary labels
                    bin_label = np.float32(abs(label) < binary)
                    binary_time_labels.append(bin_label)

                    if norm_label:
                        time_labels.append(np.float32(norm_time))
                    else:
                        time_labels.append(np.float32(label))

                    pat_ids.append(pat_id)

                    j += 1
                    if j>=self.n_pair:
                        break
        
        self.time_labels = time_labels
        self.binary_time_labels = binary_time_labels
        self.time_paths = all_eyes
        # List of patients
        self.patients = list(set(pat_ids))

    def __getitem__(self, index):
        eyes_paths = self.time_paths[index]
        eyes = []

        for file in eyes_paths:
            if file.endswith(".png"):
                eye = read_image(file)
                if self.transform:
                    seed = random.randint(0,2**32)
                    random.seed(seed)
                    torch.manual_seed(seed)
                    eye = self.transform(eye)

                eyes.append(eye)

        conv_label = self.time_labels[index]
        binary_label = self.binary_time_labels[index]

        return (eyes[0],eyes[1]), conv_label, binary_label

    def __len__(self):
        return len(self.time_labels)

class ScanDataset(Dataset):
    def __init__(self, eye_list, labels, transform=None, volume=False):
        self.volume = volume # If the volume is True read them as volumes
        super(ScanDataset,self).__init__(eye_list, labels, transform)

    def __getitem__(self, index):

        eyes_paths = self.eye_list[index]
        eyes = []

        seed = random.randint(0,2**32)
        random.seed(seed)
        torch.manual_seed(seed)
        for file in os.listdir(eyes_paths):
            if file.endswith(".png"):
                eye = read_image(eyes_paths+'/'+file)
                if self.transform:
                    eye = self.transform(eye)

                eyes.append(eye)

        conv_label = self.labels[index]

        if not self.volume:
            eyes = torch.cat(eyes, dim=0)
        elif self.volume:
            eyes = torch.stack(eyes)

        return eyes, conv_label

def ss_data_paths(data_path):
    imgs = []
    labels = []

    for root, subdirs, files in os.walk(data_path,followlinks=True):
        for file in files:
            if "central_64.png" in file:
                for j in range(-4,+5): # Arbitrary number, use 9 central B-scans from unlabeled dataset
                    labels.append(root + os.sep + file)
                    imgs.append(root + os.sep +"central_"+str(64+j*2)+".png")

    return imgs, labels

def df_to_dataset(df,path_col,label_col,transform=None):
    paths = df[path_col].tolist()
    labels = df[label_col].tolist()

    ds = Dataset(paths,labels,transform)

    return ds

class VicRegDataset(data.Dataset):
    # Makes sure that there is only 2 images from a scan in a mini-batch. This is beneficial for std loss
    # This follows the protocol from Big Self-Supervised Models Advance Medical Image Classification paper
    # Used for testing the original Vicreg
    def __init__(self, eye_list, transform=None):
        self.eye_list = eye_list
        self.transform = transform

    def __len__(self):
        return len(self.eye_list)

    def __getitem__(self, index):

        eyes_paths = self.eye_list[index]

        bscans = []
        for file in os.listdir(eyes_paths):
            if file.endswith(".png"):
                bscans.append(eyes_paths+'/'+file)

        seed = random.randint(0,2**32)
        random.seed(seed)
        sub_bscans = random.choices(bscans, k=2)

        eyes = []
        for scan in sub_bscans:
            eye = read_image(scan)
            if self.transform:
                seed = random.randint(0,2**32)
                random.seed(seed)
                torch.manual_seed(seed)
                eye = self.transform(eye)

            eyes.append(eye)

        return eyes

class VicRegTempDataset(data.Dataset):
    # This is the main dataloader for TINC
    # In a batch there is only one pair from a patient
    # In a pair, the scans are from the same position (Beware scans are not registered!)
    # This follows the protocol from Big Self-Supervised Models Advance Medical Image Classification paper
    def __init__(self, bscan_list, transform=None, binary=360, absol=False, norm_label=False, min_max=None):
        self.bscan_list = bscan_list #Bscan paths
        self.transform = transform
        self.absol = absol
        self.norm_label = norm_label
        self.min_max = min_max
        self.binary = binary

        pat_ids = []
        visit_ids = []
        bscan_ids = []
        
        for i in self.bscan_list:
            # Each path i consists of .../pat_id/visit_day/Bscan.png
            pat_ids.append(i.split('/')[-3])
            visit_ids.append(i.split('/')[-2])
            bscan_ids.append(i.split('/')[-1])

        self.pat_ids =  [x for x in list(set(pat_ids)) if sum(((x in s) and ('64' in s)) for s in self.bscan_list)>15] # ids of the patients which include at least 15 scans
        self.visit_ids = list(set(visit_ids)) # List of available dates in general
        self.bscan_ids = list(set(bscan_ids)) # Ids of the extracted  bscans in general
        
        if not self.min_max: # If the scan difference range was not
            self.min_max = (min([int(x) for x in self.visit_ids]),max([int(x) for x in self.visit_ids]))

    def __len__(self):
        return len(self.pat_ids)

    def __getitem__(self, index):

        pat_id = self.pat_ids[index]
        
        seed = random.randint(0,2**32) # Make sure different bscan positions are picked
        random.seed(seed)
        torch.manual_seed(seed)
        
        bscan = random.choice(self.bscan_ids)
        
        path1 = []
        path2 = []
        
        while not path1 or not path2: # if the current selection is not available get another scan for that patient
            dates = random.choices(self.visit_ids, k=2)
            diff = int(dates[0]) - int(dates[1])
            if abs(diff) <= self.min_max[1] and abs(diff) >= self.min_max[0]:
                path1 = [x for x in self.bscan_list if dates[0] in x and bscan in x and pat_id in x]
                path2 = [x for x in self.bscan_list if dates[1] in x and bscan in x and pat_id in x]

        path1 = path1[0] #there should be only one element in the list
        path2 = path2[0]

        eyes = []
        for scan in [path1,path2]:
            eye = read_image(scan)
            if self.transform:
                seed = random.randint(0,2**32) # Each view is transformed differently, seed to be sure
                random.seed(seed)
                torch.manual_seed(seed)
                eye = self.transform(eye)

            eyes.append(eye)
        
        label_sign = math.copysign(1,diff) # save the label of the time difference, in case the label is not absolute       
        norm_time = np.float32((abs(diff)-self.min_max[0])/(self.min_max[1]-self.min_max[0])) # min-max normalizer

        # Binary labels (redacted)
        bin_label = np.float32(abs(diff) < self.binary)

        if self.absol:
            diff = abs(diff)
        else:
            norm_time = (label_sign*(norm_time+1e-6) + 1)/2.0 # range is -1 to +1. So we add 1 -> 0-2, divide by two -> 0-1
        
        if self.norm_label:
            label = norm_time
        else:
            label = diff
        
        return (eyes[0], eyes[1]), np.float32(label), bin_label