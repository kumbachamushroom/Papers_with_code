import torch.utils.data.dataloader as dataloader
import os
import torch
import torchaudio
import numpy as np
from random import sample as Sampler
from random import shuffle as shuffle
import matplotlib.pyplot as plt
import numpy as np

PATH = "/home/lucas/PycharmProjects/Papers_with_code/data/AMI/triplet_splits"

class TripletLoader(torch.utils.data.Dataset):
    def __init__(self,base_path, anchor_positive_pairs,sample_list,train=True):
        self.base_path = base_path
        self.anchor_positive_pairs = []
        self.sample_list = []
        self.train = train
        for line in open(os.path.join(self.base_path, sample_list)):
            self.sample_list.append((line.split()[0], line.split()[1]))
        for line in open(os.path.join(self.base_path,anchor_positive_pairs)):
            negative = Sampler([sample[0] for sample in self.sample_list if sample[1] != line.split()[2]],1)
            self.anchor_positive_pairs.append((line.split()[0],line.split()[1], negative[0]))
            #print(line.split()[0], line.split()[1], negative[0])
        shuffle(self.anchor_positive_pairs)
        if self.train:
            self.anchor_positive_pairs = self.anchor_positive_pairs[0:int(0.8*len(self.anchor_positive_pairs))]
        else:
            self.anchor_positive_pairs = self.anchor_positive_pairs[int(0.8*len(self.anchor_positive_pairs)):]



    def __getitem__(self,index):
        anchor, positive, negative = str(self.anchor_positive_pairs[index][0]), str(self.anchor_positive_pairs[index][1]), str(self.anchor_positive_pairs[index][2])
        anchor, sample_rate = torchaudio.load(os.path.join(self.base_path,anchor))
        positive, sample_rate = torchaudio.load(os.path.join(self.base_path,positive))
        negative, sample_rate = torchaudio.load(os.path.join(self.base_path,negative))
        anchor_specgram = torchaudio.transforms.Spectrogram(normalized=True, power=1, n_fft=400, hop_length=100)(anchor)
        positive_specgram = torchaudio.transforms.Spectrogram(normalized=True, power=1, n_fft=400, hop_length=100)(positive)
        #negatives = [sample[0] for sample in self.sample_list if sample[1] != label]
        #random_state = np.random.RandomState(29)
        #negative = Sampler(negatives,1)
        #negative, sample_rate = torchaudio.load(os.path.join(self.base_path,negative[0]))
        negative_specgram = torchaudio.transforms.Spectrogram(normalized=True, power=1, n_fft=400, hop_length=100)(negative)
        return anchor_specgram,positive_specgram,negative_specgram

    def __len__(self):
        return len(self.anchor_positive_pairs)




class Spectrogram_Loader(torch.utils.data.Dataset):
    def __init__(self, base_path, anchor_positive_pairs, sample_list, train=True):
        self.base_path = base_path
        self.anchor_positive_pairs = anchor_positive_pairs
        self.sample_list = sample_list
        self.train = train
        self.samples = []
        for line in open(os.path.join(self.base_path, sample_list)):
            self.samples.append((line.split()[0], line.split()[1]))
        shuffle(self.samples)
        if train:
            self.samples = self.samples[0:int(0.8*len(self.samples))]
            print("TRAIN LENGTH", len(self.samples))
        else:
            self.samples = self.samples[int(0.8*len(self.samples)):]
            print("TEST LENGTH",len(self.samples))



    def __getitem__(self, index):
       sample_name, label = str(self.samples[index][0]), int(self.samples[index][1])
       track, sample_rate = torchaudio.load(os.path.join(self.base_path, sample_name))
       spectrogram = torchaudio.transforms.Spectrogram(normalized=True, power=1, n_fft=400, hop_length=100)(track)
       label = np.asarray(label)
       return spectrogram, torch.tensor(label)

    def __len__(self):
        return len(self.samples)

class Selective_Loader(torch.utils.data.Dataset):
    def __init__(self, base_path, sample_list, label, train=True, negative=True):
        self.base_path = base_path
        self.sample_list = sample_list
        self.train = train
        self.label = label
        self.samples = []
        self.negative = negative
        for line in open(os.path.join(self.base_path, sample_list)):
            self.samples.append((line.split()[0], line.split()[1]))
        if self.negative:
            self.samples = [sample[0] for sample in self.samples if sample[1] != self.label]
            print(self.samples)
            shuffle(self.samples)
        else:
            self.samples = [sample[0] for sample in self.samples if sample[1] == self.label]
            print(self.samples)
            shuffle(self.samples)

    def __getitem__(self, index):
        sample_name = str(self.samples[index])
        track , sample_rate = torchaudio.load(os.path.join(self.base_path, sample_name))
        spectrogram = torchaudio.transforms.Spectrogram(normalized=True, power=1, n_fft=400, hop_length=100)(track)
        return spectrogram

#test_loader = Negative_Loader(base_path="/home/lucas/PycharmProjects/Papers_with_code/data/AMI/triplet_splits", sample_list="sample_list.txt",label='5', train=True, negative=False)

class Frame_Loader:
    def __init__(self,path, frame_list):
        self.path = path
        self.frame_list = frame_list
        self.track, self.sample_rate = torchaudio.load(self.path)

    def __getitem__(self, index):
        start_sample = self.frame_list[index][0]*self.sample_rate
        end_sample = self.frame_list[index][1]*self.sample_rate
        spectrogram = torchaudio.transforms.Spectrogram(normalized=True, power=1, n_fft=400, hop_length=100)(self.track)
        return spectrogram

    def __len__(self):
        return len(self.frame_list)

class Triplet_Time_Loader:
    def __init__(self, path, train=True):
        self.path = path
        self.samples = []
        triplet_list = open(self.path)
        self.samples = [(line.split()[0], line.split()[1], line.split()[2], line.split()[3], line.split()[4]) for line in open(self.path)]
        shuffle(self.samples)
        shuffle(self.samples)
        if train:
            self.samples = self.samples[0:int(0.8 * len(self.samples))]
            print("TRAIN LENGTH", len(self.samples))
        else:
            self.samples = self.samples[int(0.8 * len(self.samples)):]
            print("TEST LENGTH", len(self.samples))

    def __getitem__(self, index):
        sample, string_label, int_label, start_time, stop_time = self.samples[index][0], self.samples[index][1], int(self.samples[index][2]), int(self.samples[index][3]), int(self.samples[index][4])
        track, sample_rate = torchaudio.load(sample)
        track = track[0][(start_time*sample_rate):(stop_time*sample_rate)]
        track = track.view(1, -1)
        spectrogram = torchaudio.transforms.Spectrogram(normalized=True, power=1, n_fft=400, hop_length=100)(track)
        return spectrogram, torch.tensor(int_label), string_label

    def __len__(self):
        return len(self.samples)



#test = Triplet_Time_Loader(path=os.path.join('/home/lucas/PycharmProjects/Papers_with_code/data/AMI/amicorpus_individual/Extracted_Speech','trimmed_sample_list.txt'))
#print(test.__getitem__(1))



