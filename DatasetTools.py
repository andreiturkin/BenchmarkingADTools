import numpy as np
from os import walk
from os.path import join
import csv
import re


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def GetXFromDataSet(FolderPath, Verbose=False):
    FilePath = []
    for (dirpath, _, filenames) in walk(FolderPath):
        for name in filenames:
            if (name.split('.')[0]).isdigit():
                FilePath.append(join(dirpath,name))
    FilePath.sort(key=natural_keys)
    
    Xs = []
    for i in range(len(FilePath)):
        if Verbose:
            print 'File name: {}'.format(FilePath[i])
        Xs.append(GetXFromFile(FilePath[i]))
        if Verbose:
            print 'Array length: {}\nIterator: {}\n'.format(len(Xs[i]),i+3)
        assert(i+3==len(np.array(Xs[i])))
    return Xs

def GetXFromFile(FileName):
    x = []
    with open(FileName) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row and not row[0].isspace():
                    x.append([float(str) for str in re.findall(r"[-+]?\d*\.\d+|\d+",row[0])])
    return np.array(x)
    csvfile.close()