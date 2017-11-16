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

#helper functions
def openFile(path):
    f = open(path, "r")
    return f.read()

def imshow(img,text,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

PATH = "train/taylor_swift"
dataset = LyricDataset(PATH, 2)

def convertForDict(word):
    pattern = re.compile('[\W_]+')
    word = pattern.sub('', word)
    return word.lower()

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor).cuda()

word_to_ix = {}
labels = dataset.data
for artist in labels:
    for song in os.listdir(dataset.pathToData + "/" + artist):
        if song != ".DS_Store":
            for word in openFile(dataset.pathToData + "/" + artist + "/" + song).split(" "):
                word = convertForDict(word)
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)

counter = []
loss_history = []
avg_loss = []
iteration = 0

EDIM = 512
HDIM = 512

model = SiameseLSTM(EDIM, HDIM, len(word_to_ix)).cuda()
loss = nn.MSELoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.01)
for epoch in range(5):
    for i, data in enumerate(dataset):
        song1, song2, label = data
        song1, song2 = prepare_sequence(song1, word_to_ix), prepare_sequence(song2, word_to_ix)
        label = Variable(torch.FloatTensor([label])).cuda()
        model.hidden = model.initHidden(HDIM)
        out = model(song1, song2)
        optimizer.zero_grad()
        total_loss = loss(out, label)
        total_loss.backward()
        optimizer.step()
        if i % 10 == 0 :
            print("Epoch number {}\n Current loss {}\n".format(epoch,total_loss.data[0]))
            iteration += 10
            counter.append(iteration)
            loss_history.append(total_loss.data[0])
            avg_loss.append((sum(loss_history))/len(loss_history))
        if i == 10000:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, True, filename='saved_models/checkpoint'+ str(epoch) + '.pth.tar')
            f = open("loss/loss" + str(epoch) + ".txt", "w")
            [f.write(str(l)) for l in avg_loss]
            f.close()
            break



# show_plot(counter,avg_loss)
