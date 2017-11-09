import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.autograd as autograd

import os
import numpy as np
import random
import re

class SiameseLSTM(nn.Module):
    hdim = 512
    def __init__(self, embedding_dim, vocab_size):
        super(SiameseLSTM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hdim, num_layers=2)#, bidirectional=True)
        self.hidden = self.initHidden(self.hdim)
        self.output = nn.Linear(self.hdim, 1)


    def initHidden(self, dim):
        return (autograd.Variable(torch.zeros(2, 1, dim)),
        autograd.Variable(torch.zeros(2, 1, dim)))

    def forward_once(self, lyric):
        embeds = self.embeddings(lyric).view(len(lyric), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        scores = self.output(lstm_out.view(len(lyric), -1))
        return scores

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
#         print(output1, output2)
        out1 = output1[len(output1)-1]
        out2 = output2[len(output2)-1]
#         return F.softmax(-torch.abs(out1-out2))
        return torch.abs(out1 - out2)
