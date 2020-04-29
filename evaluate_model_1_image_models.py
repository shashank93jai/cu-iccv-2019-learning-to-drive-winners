#!/usr/bin/env python
# coding: utf-8

import os
import json
import sys
import pandas as pd
from dataset import Drive360Loader
from torchvision import models
import torch.nn as nn
import torch
import numpy as np
from scipy import interpolate
import torch.optim as optim
import datetime
from itertools import islice

import pretrainedmodels
from efficientnet_pytorch import EfficientNet
import importlib.util
import yaml



save_dir = "./"
os.chdir(save_dir)
run_val = True #run a validation at the end or not
detail_size = 200 #How many batches to print results to console
output_states = list(range(0, 100)) #which epochs to save the submission file for
max_epoch = 5 #number of epochs to run
min_start_batch_idx = -1 #if restarting, what batch to restart running at, -1 means from the beginning
extrapolate = True #whether to extrapolate the data. If True but it is trained on the full dataset, reverts to False anyway
full_start = 100 # Each chapter in the submission file starts at 10s
run_time = datetime.datetime.now()

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

## Loading data from Drive360 dataset.
# load the config.json file that specifies data
# location parameters and other hyperparameters
# required.
config = json.load(open(save_dir + 'config_generate_predictions.json'))

if config['data_loader'].get('stratify_by_route'):
    # Combine all labeled data available (from the competition's train and validation splits),
    # and randomly sample chapters to form a trai/val/test split,
    # stratified on the route.
    print("Spliting original training data into test/val/test, stratifying by route.")
    train_split = .88
    val_split = .06
    
    NEW_TRAIN_FILE = './Drive360Images_160_90/drive360challenge_split_train_stratify.csv'
    NEW_VAL_FILE = './Drive360Images_160_90/drive360challenge_split_validation_stratify.csv'
    NEW_TEST_FILE = './Drive360Images_160_90/drive360challenge_split_test_stratify.csv'
    
    if os.path.exists(NEW_TRAIN_FILE):
        print('Skipping stratification of routes as csv files are already generated...')
    else:
        # load original train and validation data from the competition
        train_og = pd.read_csv('./Drive360Images_160_90/drive360challenge_train.csv')
        validation_og = pd.read_csv('./Drive360Images_160_90/drive360challenge_validation.csv')
        train_og['og_split'] = 'train'
        validation_og['og_split'] = 'validation'
        df = pd.concat([train_og, validation_og], ignore_index=True)
        del train_og
        del validation_og
        df['route'] = df['cameraRight'].apply(lambda x: x.split("/")[0])

        # chapter numbers are only unique to the original training file, so we append the 
        # original training split to differentiate unique chapters
        df['chapter'] = df.apply(lambda row: "{}_{}".format(row.chapter, row.og_split), axis=1)

        # get random chapter from each route according to the split ratios
        route_chapters = df.groupby('route')['chapter'].unique()
        route_chapters.apply(np.random.shuffle)
        splits = route_chapters.apply(lambda x: np.split(x, [int(len(x)*train_split), int(len(x)*(train_split + val_split))]))

        train_chapters = []
        val_chapters = []
        test_chapters = []
        for chapters in splits:
            train_chapters.extend(chapters[0])
            val_chapters.extend(chapters[1])
            test_chapters.extend(chapters[2])

        df[df['chapter'].isin(train_chapters)].drop('route', axis=1)\
            .reset_index(drop=True)\
            .to_csv(NEW_TRAIN_FILE, index=False)
        df[df['chapter'].isin(val_chapters)].drop('route', axis=1)\
            .reset_index(drop=True)\
            .to_csv(NEW_VAL_FILE, index=False)
        df[df['chapter'].isin(test_chapters)].drop('route', axis=1)\
            .reset_index(drop=True)\
            .to_csv(NEW_TEST_FILE, index=False)

        # Confirm that the config file is pointing to the new data splits you just created
        assert all(
            (os.path.join(config['data_loader']['data_dir'],
                          config['data_loader']['train']['csv_name']) == NEW_TRAIN_FILE,
            os.path.join(config['data_loader']['data_dir'],
                         config['data_loader']['validation']['csv_name']) == NEW_VAL_FILE,
            os.path.join(config['data_loader']['data_dir'],
                         config['data_loader']['test']['csv_name']) == NEW_TEST_FILE)),\
            "You just created data split files using stratification, "\
            "but your config file wants to use different file(s) for the data loader."

        del df

else:
    # Split the original training data into a tain/val split,
    # and use the original validation data as the test set.
    # There are 36 chapters in the original validation split.
    # We will call those the test data. Out of the 548 chapters
    # in the original training split, we will randomly partition
    # 36 of those to be the validation data and the remaining 512
    # will be the training data.
    print("Spliting original training data into test/val.")
    N_VAL_CHAPTERS = 36
    train_og = pd.read_csv('./Drive360Images_160_90/drive360challenge_train.csv')
    new_val_chapters = np.random.choice(train_og['chapter'].unique(), N_VAL_CHAPTERS, replace=False)

    train_og[train_og.chapter.isin(new_val_chapters)]\
        .reset_index(drop=True)\
        .to_csv('./Drive360Images_160_90/drive360challenge_train-split_validation.csv')
    train_og[~train_og.chapter.isin(new_val_chapters)]\
        .reset_index(drop=True)\
        .to_csv('./Drive360Images_160_90/drive360challenge_train-split_train.csv')
    
    del train_og


IMAGE_LAYER_OUTPUT_DIM = 128

# models_to_evaluate = [
#    'resnet34',  # original
#     'densenet201',
# #    'resnet50_ssl',
# #    'resnet50_swsl',
# #    'resnext50_32x4d_ssl',
# #    'resnext50_32x4d_swsl',
# #     'resnext101_32x4d',
#     'resnext101_32x4d_ssl',
#     'resnext101_32x4d_swsl',
# #     'resnext101_32x8d_swsl',  # don't use this anymore
# ]


# prepare results directory
results_path = config['results_dir']
weights_path = os.path.join(results_path, 'weights')
optimizer_path = os.path.join(results_path, 'optimizers')
for path in [results_path, weights_path, optimizer_path]:
    if not os.path.exists(path):
        os.mkdir(path)
print("Storing results data in", results_path)


#load_model_weights_path_prefix = '2020-03-26 18h48m15s_'
#def load_model_using_weights(model_key, eval=True):
#    model = SomeDrivingModel(image_model_key=model_key, feature_extract=False)
#    print('Loading model {}'.format(load_model_weights_path_prefix + model_key + '_weights.pt'))
#    model.load_state_dict(torch.load('./evaluate_model_1/'+load_model_weights_path_prefix + model_key + '_weights.pt'))
#    #model for evaluation
#    if eval:
#        model.eval()
#    return model


def set_parameter_requires_grad(model, feature_extracting):
    """from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html"""
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
            
def set_linear_layer_cadene(model, output_dim):
    """
    Set the final linear layer for models retrieved from Cadene
    """
    dim_feats = model.last_linear.in_features
    model.last_linear = nn.Linear(dim_feats, output_dim)
    return model

def set_linear_layer_ss(model, output_dim):
    """
    Set the final linear for models loaded from
    facebookresearch/semi-supervised-ImageNet1K-models
    """
    dim_feats = model.fc.in_features
    model.fc = nn.Linear(dim_feats, output_dim)
    return model


def load_ss_imagenet_models(model_key):
    return torch.hub.load(
        'facebookresearch/semi-supervised-ImageNet1K-models',
        model_key)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def set_classification_layer_simclr(model, output_dim):
    dim_feats = model.l1.in_features
    model.l1 = nn.Linear(dim_feats, output_dim)
    model.l2 = Identity()
    return model

def get_simclr_model():
    #Loading model and config file
    checkpoints_folder = 'simclr_model/checkpoints'
    config = yaml.load(open(os.path.join(checkpoints_folder, "config.yaml"), "r"))
    spec = importlib.util.spec_from_file_location("model", os.path.join(checkpoints_folder, 'resnet_simclr.py'))
    resnet_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(resnet_module)
    model = resnet_module.ResNetSimCLR(**config['model'])
    #Loading pre-trained weights
    state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model

def freeze_bottom_densenet(model):
    for param in model.features.denseblock1.parameters():
        param.requires_grad = False
    for param in model.features.denseblock2.parameters():
        param.requires_grad = False
    for param in model.features.denseblock3.parameters():
        param.requires_grad = False

def freeze_bottom_resnet(model):
    for param in model.conv1.parameters():
        param.requires_grad = False
    for param in model.bn1.parameters():
        param.requires_grad = False
    for param in model.relu.parameters():
        param.requires_grad = False
    for param in model.maxpool.parameters():
        param.requires_grad = False
    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = False
    for param in model.layer3.parameters():
        param.requires_grad = False    

            
def freeze_bottom_efficientnet(model, freeze_ratio=.75):
    """
    Freeze the bottom freeze_ratio of MBConvBlock layers in model._blocks
    (rounded to nearest int)
    """
    
    n_layers = int(np.round(
        len(model._blocks) * freeze_ratio))

    for i in range(n_layers):
        for param in model._blocks[i].parameters():
            param.requires_grad = False
            
def freeze_bottom_simclr(model):
    for param in model.features[0].parameters():
        param.requires_grad = False
    for param in model.features[1].parameters():
        param.requires_grad = False
    for param in model.features[2].parameters():
        param.requires_grad = False
    for param in model.features[3].parameters():
        param.requires_grad = False
    for param in model.features[4].parameters():
        param.requires_grad = False
    for param in model.features[5].parameters():
        param.requires_grad = False
    for param in model.features[6].parameters():
        param.requires_grad = False

def set_linear_layer_efficientnet(model, output_dim):
    dim_feats = model._fc.in_features
    model._fc = nn.Linear(dim_feats, output_dim)
    return model            

            
def get_pretrained_model(model_key, feature_extract, output_dim=128, freeze_bottom=False):
    """
    Load pretrained model for image layers.

    feature_extract: bool, whether to freeze all layers except
        the newly-added linear layer
    """

    # Models from Cadene
    if model_key == 'resnet34':
        model = pretrainedmodels.__dict__[model_key](num_classes=1000, pretrained='imagenet')  # original
        set_parameter_requires_grad(model, feature_extract)
        if freeze_bottom:
            freeze_bottom_resnet(model)
        return set_linear_layer_cadene(model, output_dim)
    elif model_key == 'resnext101_32x4d':
        model = pretrainedmodels.__dict__[model_key](num_classes=1000, pretrained='imagenet')
        model.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        set_parameter_requires_grad(model, feature_extract)
        if freeze_bottom:
            for i in range(7):
                for param in model.features[i].parameters():
                    param.requires_grad = False
        return set_linear_layer_cadene(model, output_dim)
    elif model_key == 'resnext101_64x4d':
        model = pretrainedmodels.__dict__[model_key](num_classes=1000, pretrained='imagenet')
        model.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        set_parameter_requires_grad(model, feature_extract)
        return set_linear_layer_cadene(model, output_dim)
    elif model_key == 'senet154':
        model = pretrainedmodels.__dict__[model_key](num_classes=1000, pretrained='imagenet')
        model.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        set_parameter_requires_grad(model, feature_extract)
        return set_linear_layer_cadene(model, output_dim)
    elif model_key == 'se_resnet50':
        model = pretrainedmodels.__dict__[model_key](num_classes=1000, pretrained='imagenet')
        model.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        set_parameter_requires_grad(model, feature_extract)
        return set_linear_layer_cadene(model, output_dim)
    elif model_key == 'se_resnet101':
        model = pretrainedmodels.__dict__[model_key](num_classes=1000, pretrained='imagenet')
        model.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        set_parameter_requires_grad(model, feature_extract)
        return set_linear_layer_cadene(model, output_dim)
    elif model_key == 'se_resnet152':
        model = pretrainedmodels.__dict__[model_key](num_classes=1000, pretrained='imagenet')
        model.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        set_parameter_requires_grad(model, feature_extract)
        return set_linear_layer_cadene(model, output_dim)
    elif model_key == 'se_resnext50_32x4d':
        model = pretrainedmodels.__dict__[model_key](num_classes=1000, pretrained='imagenet')
        model.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        set_parameter_requires_grad(model, feature_extract)
        return set_linear_layer_cadene(model, output_dim)
    elif model_key == 'se_resnext101_32x4d':
        model = pretrainedmodels.__dict__[model_key](num_classes=1000, pretrained='imagenet')
        model.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        set_parameter_requires_grad(model, feature_extract)
        return set_linear_layer_cadene(model, output_dim)
    elif model_key == 'densenet121':
        model = models.densenet121(pretrained=True)
        set_parameter_requires_grad(model, feature_extract)
        if freeze_bottom:
            freeze_bottom_densenet(model)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, output_dim)
        return model
    elif model_key == 'densenet161':
        model = models.densenet161(pretrained=True)
        set_parameter_requires_grad(model, feature_extract)
        if freeze_bottom:
            freeze_bottom_densenet(model)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, output_dim)
        return model
    elif model_key == 'densenet169':
        model = models.densenet169(pretrained=True)
        set_parameter_requires_grad(model, feature_extract)
        if freeze_bottom:
            freeze_bottom_densenet(model)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, output_dim)
        return model
    elif model_key == 'densenet201':
        model = models.densenet201(pretrained=True)
        set_parameter_requires_grad(model, feature_extract)
        if freeze_bottom:
            freeze_bottom_densenet(model)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, output_dim)
        return model
    elif model_key == 'resnet50_swsl':
        # from facebookresearch/semi-supervised-ImageNet1K-models:
        model = load_ss_imagenet_models(model_key)
        set_parameter_requires_grad(model, feature_extract)
        if freeze_bottom:
            freeze_bottom_resnet(model)
        return set_linear_layer_ss(model, output_dim)
    elif model_key == 'resnext50_32x4d_swsl':
        model = load_ss_imagenet_models(model_key)
        set_parameter_requires_grad(model, feature_extract)
        if freeze_bottom:
            freeze_bottom_resnet(model)
        return set_linear_layer_ss(model, output_dim)
    elif model_key == 'resnext101_32x4d_swsl':
        model = load_ss_imagenet_models(model_key)
        set_parameter_requires_grad(model, feature_extract)
        if freeze_bottom:
            freeze_bottom_resnet(model)
        return set_linear_layer_ss(model, output_dim)
    elif model_key == 'resnet50_ssl':
        model = load_ss_imagenet_models(model_key)
        set_parameter_requires_grad(model, feature_extract)
        return set_linear_layer_ss(model, output_dim)
    elif model_key == 'resnext50_32x4d_ssl':
        model = load_ss_imagenet_models(model_key)
        set_parameter_requires_grad(model, feature_extract)
        if freeze_bottom:
            freeze_bottom_resnet(model)
        return set_linear_layer_ss(model, output_dim)
    elif model_key == 'resnext101_32x4d_ssl':
        model = load_ss_imagenet_models(model_key)
        set_parameter_requires_grad(model, feature_extract)
        if freeze_bottom:
            freeze_bottom_resnet(model)
        return set_linear_layer_ss(model, output_dim)
    elif model_key == 'resnext101_32x8d_swsl':
        model = load_ss_imagenet_models(model_key)
        set_parameter_requires_grad(model, feature_extract)
        if freeze_bottom:
            freeze_bottom_resnet(model)
        return set_linear_layer_ss(model, output_dim)
    elif model_key == 'efficientnet-b0':
        model = EfficientNet.from_pretrained('efficientnet-b0')
        set_parameter_requires_grad(model, feature_extract)
        if freeze_bottom:
            freeze_bottom_efficientnet(model)
        return set_linear_layer_efficientnet(model, output_dim)
    elif model_key == 'efficientnet-b1':
        model = EfficientNet.from_pretrained('efficientnet-b1')
        set_parameter_requires_grad(model, feature_extract)
        if freeze_bottom:
            freeze_bottom_efficientnet(model)
        return set_linear_layer_efficientnet(model, output_dim)
    elif model_key == 'efficientnet-b2':
        model = EfficientNet.from_pretrained('efficientnet-b2')
        set_parameter_requires_grad(model, feature_extract)
        if freeze_bottom:
            freeze_bottom_efficientnet(model)
        return set_linear_layer_efficientnet(model, output_dim)
    elif model_key == 'efficientnet-b3':
        model = EfficientNet.from_pretrained('efficientnet-b3')
        set_parameter_requires_grad(model, feature_extract)
        if freeze_bottom:
            freeze_bottom_efficientnet(model)
        return set_linear_layer_efficientnet(model, output_dim)
    elif model_key == 'efficientnet-b4':
        model = EfficientNet.from_pretrained('efficientnet-b4')
        set_parameter_requires_grad(model, feature_extract)
        if freeze_bottom:
            freeze_bottom_efficientnet(model)
        return set_linear_layer_efficientnet(model, output_dim)
    elif model_key == 'efficientnet-b5':
        model = EfficientNet.from_pretrained('efficientnet-b5')
        set_parameter_requires_grad(model, feature_extract)
        if freeze_bottom:
            freeze_bottom_efficientnet(model)
        return set_linear_layer_efficientnet(model, output_dim)
    elif model_key == 'efficientnet-b6':
        model = EfficientNet.from_pretrained('efficientnet-b6')
        set_parameter_requires_grad(model, feature_extract)
        if freeze_bottom:
            freeze_bottom_efficientnet(model)
        return set_linear_layer_efficientnet(model, output_dim)
    elif model_key == 'efficientnet-b7':
        model = EfficientNet.from_pretrained('efficientnet-b7')
        set_parameter_requires_grad(model, feature_extract)
        if freeze_bottom:
            freeze_bottom_efficientnet(model)
        return set_linear_layer_efficientnet(model, output_dim)
    elif model_key == 'simclr':
        #simCLR model has 2 additional layers at the end which need to be removed and a final classification layer for our task needs to be added 
        model = get_simclr_model()
        set_parameter_requires_grad(model, feature_extract)
        if freeze_bottom:
            freeze_bottom_simclr(model)
        return set_classification_layer_simclr(model, output_dim)
        
        


# create a train, validation and test data loader
train_loader = Drive360Loader(config, 'train')
validation_loader = Drive360Loader(config, 'validation')
test_loader = Drive360Loader(config, 'test')
print('Length of dataset', len(test_loader.drive360.indices))


# print the data (keys) available for use. See full
# description of each data type in the documents.
print('Loaded train loader with the following data available as a dict.')
print(train_loader.drive360.dataframe.keys())


## Training a basic driving model
## The model that takes uses resnet34 and a simple 2 layer NN + LSTM architecture to predict canSteering and canSpeed.
class SomeDrivingModel(nn.Module):
    def __init__(self, image_model_key, feature_extract=True, freeze_bottom=False):
        """
        image_model: model (could be pretrained) for image layers, with the
            final linear layer already modified to be compatible with this
        """
        super(SomeDrivingModel, self).__init__()
        final_concat_size = 0
        self.image_model_key = image_model_key
        self.feature_extract = feature_extract
        self.freeze_bottom = freeze_bottom

        multiplier = 0 # count for the layers depending on config inputs

        # Main CNNs
        if config['front']: #CNN for front images
            if self.image_model_key == 'simclr':
                #Simclr model already has a relu activation applied on the last linear layer in its forward function
                self.cnn_front = get_pretrained_model(self.image_model_key, self.feature_extract, freeze_bottom=self.freeze_bottom)
            else:
                self.cnn_front = nn.Sequential(
                    get_pretrained_model(self.image_model_key, self.feature_extract, freeze_bottom=self.freeze_bottom),
                    nn.ReLU()
                )
            print("Froze {} of {} layers in cnn_front.".format(
                sum((not p.requires_grad for p in self.cnn_front.parameters())),
                len(list(self.cnn_front.parameters()))
            ))
            multiplier += 1

        if config['multi_camera']['right_left']: #CNN for side images
            if self.image_model_key == 'simclr':
                self.cnn_ls = get_pretrained_model(self.image_model_key, self.feature_extract, freeze_bottom=self.freeze_bottom)
            else:
                self.cnn_ls = nn.Sequential(
                    get_pretrained_model(self.image_model_key, self.feature_extract, freeze_bottom=self.freeze_bottom),
                    nn.ReLU()
                )
            print("Froze {} of {} layers in cnn_ls.".format(
                sum((not p.requires_grad for p in self.cnn_ls.parameters())),
                len(list(self.cnn_ls.parameters()))
            ))
            multiplier += 1
            
            if self.image_model_key == 'simclr':
                self.cnn_rs = get_pretrained_model(self.image_model_key, self.feature_extract, freeze_bottom=self.freeze_bottom)
            else:
                self.cnn_rs = nn.Sequential(
                    get_pretrained_model(self.image_model_key, self.feature_extract, freeze_bottom=self.freeze_bottom),
                    nn.ReLU()
                )
            print("Froze {} of {} layers in cnn_rs.".format(
                sum((not p.requires_grad for p in self.cnn_rs.parameters())),
                len(list(self.cnn_rs.parameters()))
            ))
            multiplier += 1

        if config['multi_camera']['rear']: #CNN for rear images
            if self.image_model_key == 'simclr':
                self.cnn_back = get_pretrained_model(self.image_model_key, self.feature_extract, freeze_bottom=self.freeze_bottom)
            else:
                self.cnn_back = nn.Sequential(
                    get_pretrained_model(self.image_model_key, self.feature_extract, freeze_bottom=self.freeze_bottom),
                    nn.ReLU()
                )
            print("Froze {} of {} layers in cnn_back.".format(
                sum((not p.requires_grad for p in self.cnn_back.parameters())),
                len(list(self.cnn_back.parameters()))
            ))
            multiplier += 1

        if config['data']: #NN for the segmentatuon map
            self.nn_data = nn.Sequential(nn.Linear(
                              20, 256), #20 input features from the map + 1 derived from location
                              nn.ReLU(),
                              nn.Linear(
                              256, 128),
                              nn.ReLU())
            multiplier += 1


        final_concat_size += 128* multiplier
        #print("Number of views is:", multiplier)

        # Main LSTM
        self.lstm = nn.LSTM(input_size=128*multiplier,
                            hidden_size=64,
                            num_layers=3,
                            batch_first=False)
        final_concat_size += 64

        # Angle Regressor
        self.control_angle = nn.Sequential(
            #Linear layers decreasing in size, with dropouts between each.
            nn.Linear(final_concat_size, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 1)
        )

        # Speed Regressor
        self.control_speed = nn.Sequential(
            #Linear layers decreasing in size, with dropouts between each.
            nn.Linear(final_concat_size, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 1)
        )

    def forward(self, data):
        module_outputs = []
        lstm_i = []

        # Loop through temporal sequence of
        # camera images and pass
        # through the cnn.
        #for k, v in data['cameraFront'].items():
        for k in data['cameraFront']:
            layers = []
            v_front = data['cameraFront'][k]
            if self.image_model_key == 'simclr':
                #Simclr's forward function returns 2 values.
                _, x = self.cnn_front(v_front)
            else:
                x = self.cnn_front(v_front) 
            layers.append(x)

            if config['data']:
                v_data = data['hereData'][k]
                y = self.nn_data(v_data)
                layers.append(y)

            if config['multi_camera']['right_left']:
                v_ls = data['cameraLeft'][k]
                if self.image_model_key == 'simclr':
                    _, y = self.cnn_ls(v_ls)
                else:
                    y = self.cnn_ls(v_ls)
                layers.append(y)

                v_rs = data['cameraRight'][k]
                if self.image_model_key == 'simclr':
                    _, z = self.cnn_rs(v_rs) 
                else:
                    z = self.cnn_rs(v_rs)
                layers.append(z)

            if config['multi_camera']['rear']:
                v_back = data['cameraRear'][k]
                if self.image_model_key == 'simclr':
                    _, u = self.cnn_back(v_back)
                else:
                    u = self.cnn_back(v_back)
                layers.append(u)

            lstm_i.append(torch.cat(layers, dim=1))
            # feed the current camera
            # output directly into the
            # regression networks.
            if k == 0:
                module_outputs.append(torch.cat(layers, dim=1))

        # Feed temporal outputs of CNN into LSTM
        i_lstm, _ = self.lstm(torch.stack(lstm_i))
        module_outputs.append(i_lstm[-1])

        # Concatenate current image CNN output
        # and LSTM output.
        x_cat = torch.cat(module_outputs, dim=-1)

        # Feed concatenated outputs into the
        # regession networks.
        prediction = {'canSteering': torch.squeeze(self.control_angle(x_cat)),
                      'canSpeed': torch.squeeze(self.control_speed(x_cat))}
        return prediction



def evaluate_model(model, phase):
    normalize_targets = config['target']['normalize']
    target_mean = config['target']['mean']
    target_std = config['target']['std']
    model.cuda()
    model.eval()
    with torch.no_grad():
        res_canSteering = []
        res_canSpeed = []
        
        loader = None
        if phase == 'val':
            loader = validation_loader
        elif phase == 'test':
            loader = test_loader
        
            for k1 in data:
                for k2 in data[k1]:
                    data[k1][k2] = data[k1][k2].cuda()
            prediction = model(data)

            if normalize_targets:
                #to change should be prediction*std_dev + mean
                res_canSpeed.extend((prediction['canSpeed'].cpu() - target['canSpeed'])*target_std['canSpeed'])
                res_canSteering.extend((prediction['canSteering'].cpu() - target['canSteering'])*target_std['canSteering'])
            else:
                res_canSpeed.extend((prediction['canSpeed'].cpu() - target['canSpeed']))
                res_canSteering.extend((prediction['canSteering'].cpu() - target['canSteering']))

            if batch_idx % detail_size == 0:
                print(batch_idx, ': ', datetime.datetime.now())

        res_canSpeed2 = np.square(res_canSpeed)
        res_canSteering2 = np.square(res_canSteering)

        print('MSE Steering: ', res_canSteering2.mean(), 'MSE Speed:', res_canSpeed2.mean())
        print('MSE Combined:', res_canSpeed2.mean() + res_canSteering2.mean())

        return {
            'mse_steering': res_canSteering2.mean().item(),
            'mse_speed':  res_canSpeed2.mean().item(),
            'mse_combined': (res_canSpeed2.mean() + res_canSteering2.mean()).item()
        }

def train_model(model, n_epochs, stop_after_batches=False):
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    for epoch in range(n_epochs):
        model.train()

        # running loss for this segment (i.e. `detail_size` batches)
        running_loss = 0.0
        running_loss_speed = 0.0
        running_loss_steering = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            if stop_after_batches and batch_idx > 2:
                break
            if batch_idx % detail_size == 0 and batch_idx > 0:
                print('Checking batch:', batch_idx)

            #Add values to the GPU
            for k1 in data:
                for k2 in data[k1]:
                    data[k1][k2] = data[k1][k2].cuda()
            for k1 in target:
                target[k1] = target[k1].cuda()

            #Forward
            optimizer.zero_grad()
            prediction = model(data)

            # Optimize on steering and speed at the same time
            loss_speed = criterion(prediction['canSpeed'], target['canSpeed'])
            loss_steering = criterion(prediction['canSteering'], target['canSteering'])
            loss = loss_speed + loss_steering
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_loss_speed += loss_speed.item()
            running_loss_steering += loss_steering.item()
            if batch_idx % detail_size == 0 and batch_idx > 0:
                print('[epoch: %d, batch:  %5d] avg. loss since last print: %.5f' %
                        (epoch, batch_idx, running_loss / detail_size), datetime.datetime.now())
                running_loss = 0.0
                running_loss_speed = 0.0
                running_loss_steering = 0.0
    return model

# Helper function to add_results to a set of lists when creating the output file.
def add_results(results, output, param=''):
    normalize_targets = config['target']['normalize']
    target_mean = config['target']['mean']
    target_std = config['target']['std']

    steering = np.squeeze(output['canSteering'].cpu().data.numpy())
    speed = np.squeeze(output['canSpeed'].cpu().data.numpy())
    if normalize_targets: #denormalize the predictions
        steering = (steering*target_std['canSteering'])+target_mean['canSteering']
        speed = (speed*target_std['canSpeed'])+target_mean['canSpeed']
    if np.isscalar(steering):
        steering = [steering]
    if np.isscalar(speed):
        speed = [speed]
        
    if param == 'speed':
        results['canSpeed'].extend(speed)
    if param == 'steering':
        results['canSteering'].extend(steering)

    
def generate_predictions(model_key, load_weights_path_speed=None, load_weights_path_steering=None):
    
    model = SomeDrivingModel(image_model_key=model_key,
                             feature_extract=False,
                             freeze_bottom=True)
    
    model.cuda()
    model.eval()
    
    results = {'canSteering': [],
               'canSpeed': []}
    
    for param in ['speed', 'steering']:
        if param == 'speed':
            model.load_state_dict(torch.load(os.path.join(weights_path, load_weights_path_speed)))
            print("Loaded model weights from", os.path.join(weights_path, load_weights_path_speed))
        else:
            model.load_state_dict(torch.load(os.path.join(weights_path, load_weights_path_steering)))
            print("Loaded model weights from", os.path.join(weights_path, load_weights_path_steering))
    
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                for k1 in data:
                    for k2 in data[k1]:
                        data[k1][k2] = data[k1][k2].cuda()
                prediction = model(data)
                add_results(results, prediction, param=param)
                if batch_idx % detail_size == 0:
                    print('Test:', batch_idx, ': ', datetime.datetime.now())

    df = pd.DataFrame.from_dict(results)
    path = os.path.join(results_path, 'results/{}.csv'.format(model_key))
    df.to_csv(path, index=False)

    
def create_submission(model, model_key, epoch=1):
    model.cuda()
    model.eval()

    results = {'canSteering': [],
               'canSpeed': []}
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            for k1 in data:
                for k2 in data[k1]:
                    data[k1][k2] = data[k1][k2].cuda()
            prediction = model(data)
            add_results(results, prediction)
            if batch_idx % detail_size == 0:
                print('Test:', batch_idx, ': ', datetime.datetime.now())

    df = pd.DataFrame.from_dict(results)
    path = os.path.join(results_path, '{:%Y-%m-%d %Hh%Mm%Ss}_{}_Submission after_epoch {}.csv'.format(run_time, model_key, epoch))
    df.to_csv(path, index=False)



all_validation_results = {}
all_test_results = {}


def train_and_val_model_epochs(model_key,
                               n_epochs,
                               load_weights_path=None,
                               load_optimizer_path=None,
                               previously_completed_epochs=0,
                               feature_extract=True,
                               freeze_bottom=False,
                               val_results_json='val_results.json',
                               test_results_json='test_results.json'):
    """
    n_epochs: how many more epochs to train (if starting from scratch, this is total epochs)
    previously_completed_epochs: if continuing to train a model. Default: 0
    """

    model = SomeDrivingModel(image_model_key=model_key,
                             feature_extract=feature_extract,
                             freeze_bottom=freeze_bottom)
    
    
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    
    if load_weights_path:
        model.load_state_dict(torch.load(os.path.join(weights_path, load_weights_path)))
        print("Loaded model weights from", os.path.join(weights_path, load_weights_path))
    
    if load_optimizer_path:
        optimizer.load_state_dict(torch.load(os.path.join(optimizer_path, load_optimizer_path)))
        print("Loaded model weights from", os.path.join(optimizer_path, load_optimizer_path))
    
    for i in range(n_epochs):
        epoch = previously_completed_epochs + i
        print(model_key+'_'+str(epoch))

        #
        # 1. Train Model
        #
        print("="*15)
        print("Training: {}, epoch {}".format(model_key, epoch), datetime.datetime.now())
        start = datetime.datetime.now()
        model.train()

        # running loss for this segment (i.e. `detail_size` batches)
        running_loss = 0.0
        running_loss_speed = 0.0
        running_loss_steering = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            # Add values to the GPU
            for k1 in data:
                for k2 in data[k1]:
                    data[k1][k2] = data[k1][k2].cuda()
            for k1 in target:
                target[k1] = target[k1].cuda()

            # Forward
            optimizer.zero_grad()
            prediction = model(data)

            # Optimize on steering and speed at the same time
            loss_speed = criterion(prediction['canSpeed'], target['canSpeed'])
            loss_steering = criterion(prediction['canSteering'], target['canSteering'])
            loss = loss_speed + loss_steering
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_loss_speed += loss_speed.item()
            running_loss_steering += loss_steering.item()
            if batch_idx % detail_size == 0 and batch_idx > 0:
                print('[epoch: %d, batch:  %5d] avg. loss since last print: %.5f' %
                      (epoch, batch_idx, running_loss / detail_size), datetime.datetime.now())
                running_loss = 0.0
                running_loss_speed = 0.0
                running_loss_steering = 0.0

        print("DONE training model:", model_key, datetime.datetime.now())
        end = datetime.datetime.now()
        torch.cuda.empty_cache()

        #
        # 2. Validation
        #
        print("Evaluating validation data:", model_key, datetime.datetime.now())
        model.eval()
        normalize_targets = config['target']['normalize']
        target_mean = config['target']['mean']
        target_std = config['target']['std']

        with torch.no_grad():
            res_canSteering = []
            res_canSpeed = []

            for batch_idx, (data, target) in enumerate(validation_loader):
                for k1 in data:
                    for k2 in data[k1]:
                        data[k1][k2] = data[k1][k2].cuda()
                prediction = model(data)

                if normalize_targets:
                    res_canSpeed.extend((prediction['canSpeed'].cpu() - target['canSpeed'])*target_std['canSpeed'])
                    res_canSteering.extend((prediction['canSteering'].cpu() - target['canSteering'])*target_std['canSteering'])
                else:
                    res_canSpeed.extend((prediction['canSpeed'].cpu() - target['canSpeed']))
                    res_canSteering.extend((prediction['canSteering'].cpu() - target['canSteering']))

                if batch_idx % detail_size == 0:
                    print(batch_idx, ': ', datetime.datetime.now())

            res_canSpeed2 = np.square(res_canSpeed)
            res_canSteering2 = np.square(res_canSteering)

            print('MSE Steering: ', res_canSteering2.mean(), 'MSE Speed:', res_canSpeed2.mean())
            print('MSE Combined:', res_canSpeed2.mean() + res_canSteering2.mean())

            validation_results_epoch = {
                'epoch': epoch,
                'mse_steering': res_canSteering2.mean().item(),
                'mse_speed':  res_canSpeed2.mean().item(),
                'mse_combined': (res_canSpeed2.mean() + res_canSteering2.mean()).item(),
                'training_minutes': (end - start) / datetime.timedelta(minutes=1)
            }
            
            all_validation_results[model_key+'_'+str(epoch)] = validation_results_epoch

            # Store validation results for this model after this training epoch
            fname = val_results_json
            with open(os.path.join(results_path, fname), 'w') as json_file:
                json.dump(all_validation_results, json_file)
            print("Stored validation results at", fname)
        
        print("DONE evaluating validation data:", model_key, datetime.datetime.now())
        torch.cuda.empty_cache()

        print("Evaluating test data:", model_key, datetime.datetime.now())
        model.eval()
        #Test results
        with torch.no_grad():
            res_canSteering = []
            res_canSpeed = []

            for batch_idx, (data, target) in enumerate(test_loader):
                for k1 in data:
                    for k2 in data[k1]:
                        data[k1][k2] = data[k1][k2].cuda()
                prediction = model(data)

                if normalize_targets:
                    res_canSpeed.extend((prediction['canSpeed'].cpu() - target['canSpeed'])*target_std['canSpeed'])
                    res_canSteering.extend((prediction['canSteering'].cpu() - target['canSteering'])*target_std['canSteering'])
                else:
                    res_canSpeed.extend((prediction['canSpeed'].cpu() - target['canSpeed']))
                    res_canSteering.extend((prediction['canSteering'].cpu() - target['canSteering']))

                if batch_idx % detail_size == 0:
                    print(batch_idx, ': ', datetime.datetime.now())

            res_canSpeed2 = np.square(res_canSpeed)
            res_canSteering2 = np.square(res_canSteering)

            print('MSE Steering: ', res_canSteering2.mean(), 'MSE Speed:', res_canSpeed2.mean())
            print('MSE Combined:', res_canSpeed2.mean() + res_canSteering2.mean())

            test_results_epoch = {
                'epoch': epoch,
                'mse_steering': res_canSteering2.mean().item(),
                'mse_speed':  res_canSpeed2.mean().item(),
                'mse_combined': (res_canSpeed2.mean() + res_canSteering2.mean()).item(),
                'training_minutes': (end - start) / datetime.timedelta(minutes=1)
            }
            all_test_results[model_key+'_'+str(epoch)] = test_results_epoch

            # Store validation results for this model after this training epoch
            fname = test_results_json
            with open(os.path.join(results_path, fname), 'w') as json_file:
                json.dump(all_test_results, json_file)
            print("Stored validation results at", fname)            
            
        print("DONE evaluating test data:", model_key, datetime.datetime.now())
        torch.cuda.empty_cache()
        
        #
        # 3. Submission file
        #
        #print("Creating submission:", model_key, datetime.datetime.now())
        #create_submission(model, model_key, epoch)
        #print("DONE creating submission", model_key, datetime.datetime.now())

        #
        # 4. Store weights
        #
        new_weights_path = os.path.join(weights_path, '{:%Y-%m-%d %Hh%Mm%Ss}_{}_weights after_epoch {}.pt'.format(run_time, model_key, epoch))
        torch.save(model.state_dict(), new_weights_path)
        print("Stored weights at", new_weights_path)
        
        new_optimizer_path = os.path.join(optimizer_path, '{:%Y-%m-%d %Hh%Mm%Ss}_{}_optimizer after_epoch {}.pt'.format(run_time, model_key, epoch))
        torch.save(optimizer.state_dict(), new_optimizer_path)
        print("Stored optimizer at", new_optimizer_path)
        print()


        
#uncomment for training related calls
model_paths = {
    'resnet34': {
        'weights': None,
        'optimizer': None,
        'prev_completed': 0,
        'n_more': 10
    },
     'densenet201': {
         'weights': None,
         'optimizer': None,
         'prev_completed': 0,
         'n_more': 10
     },
     'resnext101_32x4d_ssl': {
         'weights': None,
         'optimizer': None,
         'prev_completed': 0,
         'n_more': 10
     },
      'resnext101_32x4d_swsl': {
          'weights': None,
          'optimizer': None,
          'prev_completed': 0,
          'n_more': 10
      }
     'simclr': {
         'weights': None,
         'optimizer': None,
         'prev_completed': 0,
         'n_more': 10
     }
}

FREEZE_BOTTOM = True
FEATURE_EXTRACT = False  # If freeze_bottom = True, then this should be set to False
# n_epochs = 3  # how many more epochs to train
# previously_completed_epochs = 10  # number of epochs already completed

val_results_json = 'val_results.json'
test_results_json = 'test_results.json'

for model_key, model_config in model_paths.items():
    train_and_val_model_epochs(model_key,
                               n_epochs=model_config['n_more'],
                               load_weights_path=model_config['weights'],
                               load_optimizer_path=model_config['optimizer'],
                               previously_completed_epochs=model_config['prev_completed'],
                               feature_extract=FEATURE_EXTRACT,
                               freeze_bottom=FREEZE_BOTTOM,
                               val_results_json=val_results_json,
                               test_results_json=test_results_json)


#generating predictions on the entire test data
#model_paths = {
#   'resnet34': {
#       'weights_speed': '2020-04-15 23h42m45s_resnet34_weights after_epoch 6.pt',
#       'weights_steering': '2020-04-15 23h42m45s_resnet34_weights after_epoch 7.pt'
#    },
#    'resnext101_32x4d_ssl': {
#        'weights_speed': '2020-04-16 15h10m00s_resnext101_32x4d_ssl_weights after_epoch 15.pt',
#        'weights_steering': '2020-04-16 15h10m00s_resnext101_32x4d_ssl_weights after_epoch 18.pt'
#     },
#       'resnext101_32x4d_swsl': {
#           'weights_speed': '2020-04-15 23h42m45s_resnext101_32x4d_swsl_weights after_epoch 4.pt',
#           'weights_steering': '2020-04-16 20h42m49s_resnext101_32x4d_swsl_weights after_epoch 20.pt',
#       }
#      'simclr': {
#          'weights_speed': '2020-04-25 05h05m16s_simclr_weights after_epoch 43.pt',
#          'weights_steering': '2020-04-25 05h05m16s_simclr_weights after_epoch 40.pt',
#      }
#}
#
#for model_key, model_config in model_paths.items():
#        generate_predictions(model_key,
#                             load_weights_path_speed=model_config['weights_speed'],
#                             load_weights_path_steering=model_config['weights_steering'])




# # Train each model for 1 epoch, store validation results
# all_validation_results = {}
# STOP_AFTER_BATCHES = False  # If True, will stop training after only 3 batches -- for debugging
# FEATURE_EXTRACT = True
# models_with_errors = []
# for i, model_key in enumerate(models_to_evaluate):
#     try:
#         print("="*15)
#         print("Training:", model_key, datetime.datetime.now())
#         start = datetime.datetime.now()

#         torch.cuda.empty_cache()

#         # Train model
#         model = SomeDrivingModel(image_model_key=model_key, feature_extract=FEATURE_EXTRACT)
#         model = train_model(model, n_epochs=1, stop_after_batches=STOP_AFTER_BATCHES)

#         #Evaluate model from saved weights
#         #model = load_model_using_weights(model_key)
#         end = datetime.datetime.now()

#         torch.cuda.empty_cache()

#         # Evaluate model
#         print("Evaluating model:", model_key, datetime.datetime.now())
#         validation_results = evaluate_model(model)
#         print("DONE evaluating model:", model_key, datetime.datetime.now())

#         print("Creating submission:", model_key, datetime.datetime.now())
#         create_submission(model, model_key)
#         print("DONE creating submission", model_key, datetime.datetime.now())
#         print()

#         torch.cuda.empty_cache()
#         print("\n"*4)

#         # Store results
#         torch.save(model.state_dict(), './evaluate_model_1/{:%Y-%m-%d %Hh%Mm%Ss}_{}_weights.pt'.format(run_time, model_key))
#         all_validation_results[model_key] = validation_results
#         all_validation_results[model_key]['training_minutes'] = (end - start) / datetime.timedelta(minutes=1)
#     except Exception as e:
#         models_with_errors.append((model_key, e))
#         print("Error with model", model_key)
#         print(e)
#         print()
#     finally:
#         del model
#         torch.cuda.empty_cache()

# if len(models_with_errors) > 0:
#     print("\n"*2, ">> Models with errors:")
#     print(models_with_errors)
# else:
#     print("\n"*2, ">> No models with errors.")

# # Save validation results to file
# fname = 'val_results {:%Y-%m-%d %Hh%Mm%Ss}.json'.format(run_time)
# with open('./evaluate_model_1/' + fname, 'w') as json_file:
#     json.dump(all_validation_results, json_file)
