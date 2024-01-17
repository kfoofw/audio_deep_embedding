import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader


def normalize0_1(A):
    return (A-np.min(A))/(np.max(A) - np.min(A))

class model(nn.Module):

    def __init__(self, input_dimensions):
        self.conv1d_1 = nn.Conv1d(in_channels = input_dimensions, out_channels = 32, kernel_size = 3, stride = 1)
        self.relu_1 = nn.ReLU()
        self.maxpool1d_1 = nn.MaxPool1d(kernel_size = 2)

        self.conv1d_2 = nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1)
        self.relu_2 = nn.ReLU()
        self.maxpool1d_2 = nn.MaxPool1d(kernel_size = 2)

        self.conv1d_3 = nn.Conv1d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1)
        self.relu_3 = nn.ReLU()
        self.maxpool1d_3 = nn.MaxPool1d(kernel_size = 2)

        self.conv1d_4 = nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1)
        self.relu_4 = nn.ReLU()
        self.flatten_4 = nn.Flatten()

        self.dense_5 = nn.Linear()
        


        x = Conv1D(64, 3, strides=1, activation='relu', padding='valid', kernel_initializer='RandomNormal', bias_initializer='zeros')(x)
        x = MaxPool1D(pool_size=2, padding='valid', data_format='channels_last')(x)
        x = Conv1D(128, 3, strides=1, activation='relu', padding='valid', kernel_initializer='RandomNormal', bias_initializer='zeros')(x)
        x = MaxPool1D(pool_size=2, padding='valid', data_format='channels_last')(x)
        x = Conv1D(128, 1, strides=1, activation='relu', padding='valid', kernel_initializer='RandomNormal', bias_initializer='zeros')(x)
        x = Flatten(data_format='channels_last')(x)
        z_encoder = Dense(latent_dim, activation='linear', name='z_encoder', kernel_initializer='RandomNormal', bias_initializer='zeros')(x)
        Encoder = Model(encoder_input, z_encoder, name='Encoder')