from collections import Counter
import os
import re
from nltk.corpus import stopwords

classPaths = ['train/taylor_swift/new', 'train/taylor_swift/old']

def convertForDict(word):
    pattern = re.compile('[\W_]+')
    word = pattern.sub('', word)
    return word.lower()

for path in classPaths:
    counts = Counter()
    text = ''
    for song in os.listdir(path):
        if song != '.DS_Store':
            lyrics = open(path + '/' + song, 'r').read()
            print(lyrics.split(" "))
            for word in lyrics.split(" "):
                text += convertForDict(word)
                text += " "
    s = Counter(text.split(" "))
    print(s)
