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
import shutil

from LyricDataset import LyricDataset
from SiameseLSTM import SiameseLSTM
from LyricNet import prepare_sequence

def loadModel(path):
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}'".format(path))

model = loadModel("saved_models/checkpoint9.pth.tar")

TEST_PATH = "test"
testset = LyricDataset(TEST_PATH, 2)

for i, data in enumerate(testset,0):
    song1, song2, label = data
    song1, song2 = prepare_sequence(song1, word_to_ix, True), prepare_sequence(song2, word_to_ix, True)
    song1, song2 = song1.cuda(), song2.cuda()
    label = Variable(torch.FloatTensor([label]).cuda())
    model.hidden = model.initHidden(HDIM)
    out = model(song1, song2)
    print(label)
    print(out)
    if i == 0:
        break
