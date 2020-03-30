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

np.random.seed(88)
torch.manual_seed(88)

## Loading data from Drive360 dataset.
# load the config.json file that specifies data
# location parameters and other hyperparameters
# required.
config = json.load(open(save_dir + 'evaluate_config_1.json'))


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

train_og[train_og.chapter.isin(new_val_chapters)].reset_index(drop=True).to_csv('./Drive360Images_160_90/drive360challenge_train-split_validation.csv')
train_og[~train_og.chapter.isin(new_val_chapters)].reset_index(drop=True).to_csv('./Drive360Images_160_90/drive360challenge_train-split_train.csv')



IMAGE_LAYER_OUTPUT_DIM = 128

models_to_evaluate = [
    'resnet34',  # original
#    'resnext101_32x4d',
#     'resnext101_64x4d', # Too big
    'senet154',  # Too big for GPU. Try rewriting training loop to be more memory-efficient
    'se_resnet50',
    'se_resnet101',
    'se_resnet152',
    'se_resnext50_32x4d',
    'se_resnext101_32x4d',
    'densenet121',
    'densenet161',
    'densenet169',
    'densenet201',
    'resnet50_swsl',
    'resnext50_32x4d_swsl',
    'resnext101_32x4d_swsl',
    'resnet50_ssl',
    'resnext50_32x4d_ssl',
    'resnext101_32x4d_ssl'
]

#def load_model_using_weights(model_key, eval=True):
#    model = SomeDrivingModel(image_model_key=model_key, feature_extract=False)
#    print('Loading model {}'.format(load_model_weights_path_prefix + model_key + '_weights.pt'))
#    model.load_state_dict(torch.load('./evaluate_model_1/'+load_model_weights_path_prefix + model_key + '_weights.pt'))
     #model for evaluation
#   if eval:
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

def get_pretrained_model(model_key, feature_extract, output_dim=128):
    """
    Load pretrained model for image layers.

    feature_extract: bool, whether to freeze all layers except
        the newly-added linear layer
    """

    # Models from Cadene
    if model_key == 'resnet34':
        model = pretrainedmodels.__dict__[model_key](num_classes=1000, pretrained='imagenet')  # original
        set_parameter_requires_grad(model, feature_extract)
        return set_linear_layer_cadene(model, output_dim)
    elif model_key == 'resnext101_32x4d':
        model = pretrainedmodels.__dict__[model_key](num_classes=1000, pretrained='imagenet')
        model.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        set_parameter_requires_grad(model, feature_extract)
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
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, output_dim)
        return model
    elif model_key == 'densenet161':
        model = models.densenet161(pretrained=True)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, output_dim)
        return model
    elif model_key == 'densenet169':
        model = models.densenet169(pretrained=True)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, output_dim)
        return model
    elif model_key == 'densenet201':
        model = models.densenet201(pretrained=True)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, output_dim)
        return model
    elif model_key == 'resnet50_swsl':
        # from facebookresearch/semi-supervised-ImageNet1K-models:
        model = load_ss_imagenet_models(model_key)
        set_parameter_requires_grad(model, feature_extract)
        return set_linear_layer_ss(model, output_dim)
    elif model_key == 'resnext50_32x4d_swsl':
        model = load_ss_imagenet_models(model_key)
        set_parameter_requires_grad(model, feature_extract)
        return set_linear_layer_ss(model, output_dim)
    elif model_key == 'resnext101_32x4d_swsl':
        model = load_ss_imagenet_models(model_key)
        set_parameter_requires_grad(model, feature_extract)
        return set_linear_layer_ss(model, output_dim)
    elif model_key == 'resnet50_ssl':
        model = load_ss_imagenet_models(model_key)
        set_parameter_requires_grad(model, feature_extract)
        return set_linear_layer_ss(model, output_dim)
    elif model_key == 'resnext50_32x4d_ssl':
        model = load_ss_imagenet_models(model_key)
        set_parameter_requires_grad(model, feature_extract)
        return set_linear_layer_ss(model, output_dim)
    elif model_key == 'resnext101_32x4d_ssl':
        model = load_ss_imagenet_models(model_key)
        set_parameter_requires_grad(model, feature_extract)
        return set_linear_layer_ss(model, output_dim)


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
    def __init__(self, image_model_key, feature_extract=True):
        """
        image_model: model (could be pretrained) for image layers, with the
            final linear layer already modified to be compatible with this
        """
        super(SomeDrivingModel, self).__init__()
        final_concat_size = 0
        self.image_model_key = image_model_key
        self.feature_extract = feature_extract

        multiplier = 0 # count for the layers depending on config inputs

        # Main CNNs
        if config['front']: #CNN for front images
            self.cnn_front = nn.Sequential(
                get_pretrained_model(self.image_model_key, self.feature_extract),
                nn.ReLU()
            )
            multiplier += 1

        if config['multi_camera']['right_left']: #CNN for side images
            self.cnn_ls = nn.Sequential(
                get_pretrained_model(self.image_model_key, self.feature_extract),
                nn.ReLU()
            )
            multiplier += 1

            self.cnn_rs = nn.Sequential(
                get_pretrained_model(self.image_model_key, self.feature_extract),
                nn.ReLU()
            )
            multiplier += 1

        if config['multi_camera']['rear']: #CNN for rear images
            self.cnn_back = nn.Sequential(
                get_pretrained_model(self.image_model_key, self.feature_extract),
                nn.ReLU()
            )
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
            x = self.cnn_front(v_front)
            layers.append(x)

            if config['data']:
                v_data = data['hereData'][k]
                y = self.nn_data(v_data)
                layers.append(y)

            if config['multi_camera']['right_left']:
                v_ls = data['cameraLeft'][k]
                y = self.cnn_ls(v_ls)
                layers.append(y)

                v_rs = data['cameraRight'][k]
                z = self.cnn_ls(v_rs)
                layers.append(z)

            if config['multi_camera']['rear']:
                v_back = data['cameraRear'][k]
                u = self.cnn_ls(v_back)
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



def evaluate_model(model):
    normalize_targets = config['target']['normalize']
    target_mean = config['target']['mean']
    target_std = config['target']['std']
    model.cuda()
    model.eval()
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
def add_results(results, output):
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
    results['canSteering'].extend(steering)
    results['canSpeed'].extend(speed)

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
    df.to_csv('./evaluate_model_1/{:%Y-%m-%d %Hh%Mm%Ss}_{}_Submission after_epoch {}.csv'.format(run_time, model_key, epoch), index=False)


def train_and_val_model_epochs(model_key,
                               n_epochs,
                               weights_path_template=None,
                               optimizer_path_template=None,
                               previously_completed_epochs=0,
                               feature_extract=True):
    """
    n_epochs: how many more epochs to train (if starting from scratch, this is total epochs)
    weights_path_template: if loading model weights, this is string template with place
        for model key. E.g. "2020-03-26 14h14m12s_{}_weights.pt"
    previously_completed_epochs: if continuing to train a model. Default: 0
    """

    model = SomeDrivingModel(image_model_key=model_key, feature_extract=feature_extract)
    if weights_path_template:
        # load from path
        model.load_state_dict(torch.load(weights_path_template.format(model_key)))

    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    
    if optimizer_path_template:
        optimizer.load_state_dict(torch.load(optimizer_path_template.format(model_key)))

    for i in range(n_epochs):
        epoch = previously_completed_epochs + i + 1

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
        print("Evaluating model:", model_key, datetime.datetime.now())
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
                'mse_steering': res_canSteering2.mean().item(),
                'mse_speed':  res_canSpeed2.mean().item(),
                'mse_combined': (res_canSpeed2.mean() + res_canSteering2.mean()).item(),
                'training_minutes': (end - start) / datetime.timedelta(minutes=1)
            }

            # Store validation results for this model after this training epoch
            fname = 'val_results {} after_epoch {} {:%Y-%m-%d %Hh%Mm%Ss}.json'.format(model_key, epoch, run_time)
            with open('./evaluate_model_1/' + fname, 'w') as json_file:
                json.dump(validation_results_epoch, json_file)
            print("Stored validation results at", fname)

        print("DONE evaluating model:", model_key, datetime.datetime.now())
        torch.cuda.empty_cache()

        #
        # 3. Submission file
        #
        print("Creating submission:", model_key, datetime.datetime.now())
        create_submission(model, model_key, epoch)
        print("DONE creating submission", model_key, datetime.datetime.now())

        #
        # 4. Store weights
        #
        new_weights_path = './evaluate_model_1/{:%Y-%m-%d %Hh%Mm%Ss}_{}_weights after_epoch {}.pt'.format(run_time, model_key, epoch)
        torch.save(model.state_dict(), new_weights_path)
        print("Stored weights at", new_weights_path)
        
        new_optimizer_path = './evaluate_model_1/{:%Y-%m-%d %Hh%Mm%Ss}_{}_optimizer after_epoch {}.pt'.format(run_time, model_key, epoch)
        torch.save(optimizer.state_dict(), new_optimizer_path)
        print("Stored optimizer at", new_optimizer_path)
        print()


FEATURE_EXTRACT = False
n_epochs = 2  # how many more epochs to train - if you want to resume training from a checkpoint
weights_path_template = "./evaluate_model_1/2020-03-26 18h48m15s_{}_weights.pt"
previously_completed_epochs = 1

for model_key in models_to_evaluate:
    train_and_val_model_epochs(model_key,
                               n_epochs,
                               weights_path_template=weights_path_template,
                               optimizer_path_template=None,
                               previously_completed_epochs=previously_completed_epochs,
                               feature_extract=FEATURE_EXTRACT)



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
