import cv2
import numpy as np 
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from matplotlib import pyplot as plt
import albumentations.pytorch as AT
# from info import AllLabel
from sklearn.metrics import roc_curve, auc
from albumentations import (
    RandomRotate90, Flip, Transpose, GaussNoise, Blur, VerticalFlip, HorizontalFlip, Flip, \
    HueSaturationValue, RGBShift, RandomBrightness, Resize, Normalize, Compose, CenterCrop, \
    GaussianBlur,Cutout,RandomShadow, RandomContrast, RandomBrightnessContrast, ShiftScaleRotate, \
    ElasticTransform, RandomGridShuffle, OneOf)

# from model import EfficientNet


def transforms(size_image):
    return Compose([
        Resize(size_image, size_image),
        Normalize(mean=(0.591, 0.621, 0.806), std=(0.117, 0.105, 0.092)),
        AT.ToTensor()
    ])

# base = AllLabel()
# dataroot = '/datapath/train_resize/'
dataroot = 'E:/SIIM/test/'

def extractData(csv_path, json_path = ''):
    # extract data from csv file
    df = pd.load_csv(csv_path)
    row, column = df.shape
    dict4json = dict()

    # # Open this part to get information for info.py
    # # original data
    # sex = df.sex
    # age = df.age_approx
    # location = df.anatom_site_general_challenge
    # label = df.target

    # # convert to our form
    # sex = set(sex)
    # age = age.unique().astype(np.int32)
    # location = set(location)
    # label = set(label)

    sex = base.sex()
    age = base.age()
    location = base.location()
    
    for idx in range(row):
        line = df.iloc[idx]
        name = line.image_name
        if line.sex not in sex:
            line.sex = 'Not sure'
        this_sex = int(np.argwhere(line.sex))
        this_age = int(np.argwhere(line.age_approx.astype(np.int32)))
        if line.location not in location:
            line.location = 'empty'
        this_location = int(np.argwhere(line.location))
        dict4json[name] = {'sex': this_sex,
                           'age': this_age,
                           'location': this_location,
                           'label': label}
    
    # # Open this part if need to dump the label to json file
    if json_path:
        import json
        with open(json_path, 'w') as f:
            json.dump(dict4json,f)

        label = int(line.label)
    return dict4json

def evaluate(y_true, predict_score, save_path, pos_label = 1, plot = True):
    # AUC score for model
    fpr, tpr, thresholds = roc_curve(y_true, predict_score, pos_label = 1)
    if save_path:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig(save_path)
        if plot:
            plt.show()
    return auc(fpr, tpr)

def confusion_matrix(y_true, predict_score, save_path):
    import sklearn.metrics as sk
    import seaborn as sn
    y_true = np.reshape(y_true, [max(y_true.shape)])
    predict_score = np.reshape(predict_score, [max(predict_score.shape)])
    predict = predict_score > 0.5
    confusion = np.round(sk.confusion_matrix(y_true,predict))
    df_cm = pd.DataFrame(confusion, index = [i for i in ["Benign", "Malignant"]],
                    columns = [i for i in ["Benign", "Malignant"]])
    plt.figure(figsize = (10,8))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 20}, fmt="d")
    plt.savefig(save_path)


def augmentation(size_image,p=0.5):
    return Compose([
        Resize(size_image, size_image),
        RandomRotate90(),
        Flip(),
        Transpose(),
        GaussNoise(),
        Blur(),
        VerticalFlip(),
        HorizontalFlip(),
        HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=10),
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
        RandomBrightness(limit = 0.05),
        CenterCrop(height = 150, width = 150, p = 0.5)    
    ], p=p)

def augmentation_hardcore(size_image, p = 0.8):
    '''
    Only use for second model
    About albumentation, p in compose mean the prob that all transform in Compose work
    '''
    return Compose([
        Resize(size_image, size_image),
        CenterCrop(height = 200, width = 200, p = 0.5),
        Cutout(),
        RandomShadow(shadow_dimension = 3),
        OneOf([
            Flip(),
            VerticalFlip(),
            HorizontalFlip(),
        ], p = 0.5),
        OneOf([
            RandomRotate90(),
            Transpose(),
        ], p = 0.5),
        OneOf([
            GaussNoise(),
            GaussianBlur(blur_limit = 9),
            Blur()
        ], p = 0.5),
        OneOf([
            HueSaturationValue(hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=20),
            RGBShift(),
            RandomBrightness(brightness_limit = 0.4),
            RandomContrast(),
            RandomBrightnessContrast(),
        ], p = 0.5),
        OneOf([
            ShiftScaleRotate(),
            ElasticTransform(),
            RandomGridShuffle()
        ], p = 0.5)           
    ], p = p)

class PyTorchImageDataset(Dataset):
    def __init__(self, image_list, labels,train, size_image, **kwags):
        self.image_list = image_list
        self.labels = labels
        self.transforms = transforms(size_image)
        self.train = train
        self.hard = kwags.pop('hard', False)
        # if self.hard:
        #     self.augment = augmentation_hardcore(size_image)
        # else:
        #     self.augment = augmentation(size_image)
        self.augment_h = augmentation_hardcore(size_image)
        self.augment_s = augmentation(size_image)
    def __len__(self):
        # return (len(self.image_list))
        return 32
    
    def __getitem__(self, i):
        image = cv2.imread(dataroot + self.image_list[i] + '.jpg')
        label = self.labels[i]
        if self.train:
            # if self.hard and label == 0:
            #     pass
            # else:    
            #     image = self.augment(image= image)['image']
            if self.hard and label == 0:
                image = self.augment_s(image = image)['image']
            elif self.hard and label == 1:
                image = self.augment_h(image = image)['image']
            else:
                image = self.augment_s(image = image)['image']
            
        image = self.transforms(image = image)['image']
            
        return image, label


class PyTorchTestImageDataset(Dataset):
    def __init__(self, image_list, size_image):
        self.image_list = image_list
        self.transforms = transforms(size_image)
        
    def __len__(self):
        return (len(self.image_list))
    
    def __getitem__(self, i):
        image = cv2.imread(dataroot + self.image_list[i] + '.jpg')
        image = self.transforms(image = image)['image']
        return image

class TobyNet():
    def __init__(self, model_name, weights_path, **kwags):
        self.model_name = model_name
        self.weights_path = weights_path
        self.num_classes = kwags.pop('num_classes', 1000)
        self.imagenet = kwags.pop('imagenet', False)

    def initModel(self):
        try:
            model = EfficientNet.from_pretrained(self.model_name, weights_path= self.weights_path, num_classes= self.num_classes)
            return model
        except TypeError:
            model = EfficientNet.from_pretrained(self.model_name, weights_path= self.weights_path)
            model._fc = nn.Linear(model._fc.in_features, self.num_classes)
            return model

        
def freezeResNetFC(model):
    # can't use: some parameters don't need to gradient
    for name, p in model.named_parameters():
        # if not ("fc" in name):
          p.requires_grad = False
            
def unfreezeResNetFC(model):
    for param in model.parameters():
        param.requires_grad = True

def oversamplingData(X,y, labels_less = 1):
    '''
    Duplicate number of sample which was imbalanced
    '''
    X = np.array(X)
    y = np.array(y)
    position_less = np.where(y==labels_less)
    num_less = len(position_less[0])
    num_more = len(X) - num_less
    scale = num_more//num_less
    scale = scale -1 if scale > 2 else 1

    # suprise?
    scale = 30

    less_sample = np.repeat(X[position_less], scale) 
    less_label = np.repeat([labels_less], len(less_sample))
    X = np.concatenate((X,less_sample))
    y = np.concatenate((y,less_label))
    return X, y 

def extractPosNeg(y):
    pos = np.where(y==1)
    neg = np.where(y==0)
    return len(pos[0]), len(neg[0])