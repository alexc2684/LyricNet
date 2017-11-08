import os

class LyricDataset(Dataset):
    def __init__(self, pathToData, numClasses, should_invert=True):
        self.pathToData = pathToData
        self.data = [path for path in os.listdir(self.pathToData) if os.path.isdir(self.pathToData + "/" + path)]
        self.should_invert = should_invert
        self.numClasses = numClasses

    def __getitem__(self, index):
        labels = self.data
        sameClass = random.randint(0,1)
        song1 = None
        song2 = None
        if sameClass:
            label = random.randint(0, self.numClasses-1)
            classPath = self.pathToData + "/" + labels[label]
            songs = [path for path in os.listdir(classPath) if not path.startswith(".")]
            index1 = random.randint(0,len(songs)-1)
            s1Path = classPath + "/" + songs[index1]
            index2 = index1
            while index2 == index1:
                index2 = random.randint(0,len(songs)-1)
            s2Path = classPath + "/" + songs[index2]
#             print(songs[index1], songs[index2])

        else:
            label1 = random.randint(0, self.numClasses-1)
            label2 = label1
            while label2 == label1:
                label2 = random.randint(0, self.numClasses-1)
            classPath1 = self.pathToData + "/" + labels[label1]
            songs1 = [path for path in os.listdir(classPath1) if not path.startswith(".")]
            index1 = random.randint(0,len(songs1)-1)
            s1Path = classPath1 + "/" + songs1[index1]
            classPath2 = self.pathToData + "/" + labels[label2]
            songs2 = [path for path in os.listdir(classPath2) if not path.startswith(".")]
            index2 = random.randint(0,len(songs2)-1)
            s2Path = classPath2 + "/" + songs2[index2]
#             print(songs1[index1], songs2[index2])
        s1 = openFile(s1Path)
        lyric1 = [convertForDict(word) for word in s1.split(" ")]
        s2 = openFile(s2Path)
        lyric2 = [convertForDict(word) for word in s2.split(" ")]
        return lyric1, lyric2, sameClass

    def __len__(self):
        return sum([len(os.listdir(self.pathToData + "/" + p)) for p in os.listdir(self.pathToData) if p != ".DS_Store"])
