from __future__ import print_function
import torch
import torch.nn as nn
from Triplet_DataLoader import TripletLoader
from utlis import get_conv2d_output_shape, get_conv1d_output_shape
import os
import numpy as np
import wandb
from Triplet_Net import VGGVox, TripletNet
import argparse
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from visdom import Visdom
from torch.autograd import Variable
from Triplet_DataLoader import Selective_Loader
from losses import batch_hard_triplet_loss
import numpy as np
import time
from Prepare_Track import Prepare_Track
import pandas as pd
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torchaudio
import random
from Triplet_DataLoader import Triplet_Time_Loader

#def compare_triplets_by_dist(negative_loader,positive_loader,margin,model):


def Plot_Results(df_true,df_model):
    speaker_labels = df_true.index.values
    for speaker in speaker_labels:
        fig,ax  = plt.subplots()
        labels = df_true.loc[speaker,:].values[0:1000]
        timestamps = np.where(labels == 1)
        labels = labels[labels == 1]
        plt.plot(timestamps[0], labels, 'o', label=speaker+'_truth')

        labels = df_model.loc[speaker,:].values[0:1000]
        timestamps = np.where(labels == 1)
        labels = labels[labels == 1]
        plt.plot(timestamps[0], labels*2, 'o', color='red', label=speaker+'_model')
        plt.legend('ground truth', ' model ')
        plt.show()






def get_target_embeddings(num_embeddings, label_df, sample_list, model):
    model.eval()
    speaker_labels = label_df.index.values
    f = [(line.split()[0], line.split()[1], line.split()[2], line.split()[3], line.split()[4]) for line in open(sample_list)]
    Avg_Embedding_List = []
    for speaker in speaker_labels:
        print('Finding ',speaker)
        samples = [sample for sample in f if sample[1] == speaker]
        samples = random.sample(samples,num_embeddings)
        embeddingList = []
        for file in samples:
            track, sample_rate = torchaudio.load(file[0])
            track = track[0][(int(file[3])*sample_rate):(int(file[4])*sample_rate)]
            track = track.view(1,-1)
            spectrogram = torchaudio.transforms.Spectrogram(normalized=True, power=1, n_fft=400, hop_length=100)(track).view(1,1,201,481).cuda()
            embedding = model(spectrogram)
            embeddingList.append(embedding)
        embeddingList = torch.stack(embeddingList).view(len(embeddingList),1,256)
        mean = torch.mean(embeddingList, dim=0, keepdim=True)
        Avg_Embedding_List.append((mean, speaker))
    return Avg_Embedding_List

def diarize(avg_embeddings, frame_list, label_df, model, path, margin=20):
    model.eval()
    track, sample_rate = torchaudio.load(path)
    model_labels = pd.DataFrame(index=label_df.index.values, columns = label_df.columns.values)
    for frame in label_df.columns.values:
        start = int(frame_list[int(frame)][0])
        stop = int(frame_list[int(frame)][1])
        snippet = track[0][(start*sample_rate):(stop*sample_rate)].view(1,-1)
        spectrogram = torchaudio.transforms.Spectrogram(normalized=True, power=1, n_fft=400, hop_length=100)(snippet).view(1,1,201,481).cuda()
        embedding = model(spectrogram)
        for speaker in avg_embeddings:
            if torch.dist(speaker[0], embedding) < margin:
                model_labels.loc[speaker[1],frame] = 1
        print(" Scoring frame {}/{} ".format(frame, len(frame_list)), end='\r', flush=True)
    model_labels.fillna(0)
    return model_labels










def get_average_embedding(window_size, num_embeddings, path, model):
    track, sample_rate = torchaudio.load(path)
    window_size = sample_rate*window_size
    sampler_start = random.randint(0,len(track[0])-num_embeddings*window_size)
    embeddingList = []
    for i in range(num_embeddings):
        sample = track[0][sampler_start:(sampler_start+window_size)]
        sample = sample.view(1,-1)
        spectrogram = torchaudio.transforms.Spectrogram(normalized=True, power=1, n_fft=400, hop_length=100)(sample)
        spectrogram = spectrogram.cuda()
        #print(spectrogram.size())
        spectrogram = spectrogram.view(1,1,201,481)
        sampler_start = sampler_start+window_size
        embedding = model(spectrogram)
        embeddingList.append(embedding)
    #print(embedding.size())
    #print(embeddingList)
    embeddingList = torch.stack(embeddingList).view(len(embeddingList),1,256)
    #print(embeddingList.size())
    mean = torch.mean(embeddingList, dim=0,keepdim=True)
    #mean = embeddingList[0]
    return mean

def compare_embeddings(frame_list,label_df,model,path,speaker_label,anchor, margin=0.45):
    model.eval()
    track, sample_rate = torchaudio.load(path)
    frame_scores = np.zeros_like(label_df.loc[speaker_label,:].values)
    true_labels = label_df.loc[speaker_label,:].values
    for i, frame in enumerate(frame_list):
        start = int(frame[0])
        stop = int(frame[1])
        snippet = track[0][(start*sample_rate):(stop*sample_rate)]
        snippet = snippet.view(1,-1)
        spectrogram = torchaudio.transforms.Spectrogram(normalized=True, power=1, n_fft=400,hop_length=100)(snippet)
        spectrogram = spectrogram.cuda()
        spectrogram = spectrogram.view(1,1,201,481)
        embedding = model(spectrogram)
        frame_scores[i] = 1 if torch.dist(anchor,embedding,p=2) < 20 else 0
        print(" Scoring frame {}/{} ".format(i,len(frame_list)), end='\r', flush=True)
    result = frame_scores*label_df.loc[speaker_label,:].values
    print(result)
    accuracy = (np.count_nonzero(result)/np.count_nonzero(label_df.loc[speaker_label,:].values))*100
    print(accuracy)

    print('done')
    return result

def compare_track(model):
    model.eval()
    track1 = '/home/lucas/PycharmProjects/Papers_with_code/data/AMI/triplet_splits/EN2001a_Speaker_A_12.wav'
    track2 = '/home/lucas/PycharmProjects/Papers_with_code/data/AMI/triplet_splits/EN2001a_Speaker_A_46.wav'
    track3 = '/home/lucas/PycharmProjects/Papers_with_code/data/AMI/triplet_splits/EN2001a_Speaker_B_29.wav'
    track1, sample_rate = torchaudio.load(track1)
    track2, sample_rate = torchaudio.load(track2)
    track3, sample_rate = torchaudio.load(track3)
    spectrogram = torchaudio.transforms.Spectrogram(normalized=True, power=1, n_fft=400, hop_length=100)(track1)
    spectrogram = spectrogram.view(1, 1, 201, 481)
    embedding1 = model(spectrogram.cuda())
    spectrogram = torchaudio.transforms.Spectrogram(normalized=True, power=1, n_fft=400, hop_length=100)(track2)
    spectrogram = spectrogram.view(1, 1, 201, 481)
    embedding2 = model(spectrogram.cuda())
    spectrogram = torchaudio.transforms.Spectrogram(normalized=True, power=1, n_fft=400, hop_length=100)(track3)
    spectrogram = spectrogram.view(1, 1, 201, 481)
    embedding3 = model(spectrogram.cuda())
    print(torch.dist(embedding1, embedding2, p=2))
    print(torch.dist(embedding1,embedding3,p=2))












def main():
    #Train settings
    wandb.init(project="vgg_triplet")
    global args, best_acc
    parser = argparse.ArgumentParser(description='VGG Triplet-Loss Speaker Embedding')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--no-cuda', action='store_true',default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed')
    parser.add_argument('--log-interval',type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training score')
    parser.add_argument('--margin', type=float, default=1, metavar='M',
                        help='margin for triplet loss (default: 0.2)')
    parser.add_argument('--base-path',type=str,
                        default='/home/lucas/PycharmProjects/Papers_with_code/data/AMI/triplet_splits',
                        help='string to triplets')
    parser.add_argument('--s-file',default='/home/lucas/PycharmProjects/Papers_with_code/data/AMI/amicorpus_individual/Extracted_Speech/trimmed_sample_list.txt',type=str,
                        help='name of sample list')
    parser.add_argument('--load-path',default='/home/lucas/PycharmProjects/Papers_with_code/data/models/VGG_Triplet'
                        ,type=str, help='path to save models to')
    parser.add_argument('--name', default='Compare_Embeddings', type=str,
                        help='name of experiment')
    parser.add_argument('--base-sample', default='/home/lucas/PycharmProjects/Papers_with_code/data/AMI/amicorpus_individual/Extracted_Speech/EN2001d_MEO069.wav', type=str,
                        help='path to base sample')
    parser.add_argument('--test-path', type=str, default='/home/lucas/PycharmProjects/Papers_with_code/data/AMI/pyannote/amicorpus/EN2001a/audio/EN2001a.Mix-Headset.wav'
                        ,help='path to audio file to be used in testing')

    args = parser.parse_args()

    #wandb.run.name = args.name
    #wandb.run.save()
    #wandb.config.update(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args.no_cuda, torch.cuda.is_available())
    if args.cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    torch.manual_seed(args.seed)
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

    ground_truth = Prepare_Track(track_name='EN2001a.Mix-Headset',rttm_name='MixHeadset.train.rttm',path_to_track='/home/lucas/PycharmProjects/Papers_with_code/data/AMI/pyannote/amicorpus/EN2001a/audio', path_to_rttm='/home/lucas/PycharmProjects/Papers_with_code/data/AMI/pyannote/AMI')
    true_label_df, true_frame_list, speaker_dict = ground_truth.label_frames(window_size=3, step_size=0.1)

    model = VGGVox()
    model.to(device)
    model.load_state_dict(torch.load(args.load_path))
    print("Model loaded from state dict")
    cudnn.benchmark = True

    #print(get_target_embeddings(num_embeddings=3,label_df=true_label_df,sample_list=args.s_file,model=model))
    anchor = get_average_embedding(window_size=3, num_embeddings=5, path=args.base_sample, model=model)
    #print(diarize(get_target_embeddings(num_embeddings=3, label_df=true_label_df, sample_list=args.s_file, model=model), frame_list=true_frame_list,label_df=true_label_df, model=model,path=args.test_path,margin=20))

    speaker_label = args.base_sample[args.base_sample.rfind('_')+1:args.base_sample.rfind('.')]
    result = compare_embeddings(frame_list=true_frame_list, label_df=true_label_df, model=model, path=args.test_path,anchor=anchor,speaker_label=speaker_label,
                       margin=0.45)
    #compare_track(model)
    #Plot_Results(true_label_df, df_model=diarize(get_target_embeddings(num_embeddings=1, label_df=true_label_df, sample_list=args.s_file, model=model), frame_list=true_frame_list,label_df=true_label_df, model=model,path=args.test_path,margin=20))


    #negative_loader = torch.utils.data.DataLoader(Selective_Loader(base_path=args.base_path, sample_list=args.s_file,label='5',train=True,negative=True))
    #positive_loader = torch.utils.data.DataLoader(Selective_Loader(base_path=args.base_path, sample_list=args.s_file, label='5', train=True, negative=False))

    #train_time_loader = torch.utils.data.DataLoader(
    #    Triplet_Time_Loader(path=os.path.join('/home/lucas/PycharmProjects/Papers_with_code/data/AMI/amicorpus_individual/Extracted_Speech', 'trimmed_sample_list.txt'), train=True), batch_size=16,
    #    shuffle=True, **kwargs)
    #i = 0
    #model.eval()
    #for batch in iter(train_time_loader):
    #    if i < 1:
    #        spectrograms, int_labels, string_labels = batch
    #        spectrograms = spectrograms.cuda()
    #        print(spectrograms.size())
    #        embeddings = model(spectrograms)
    #        print(embeddings)
    #        i = i + 1
    #    else:
    #        return




if __name__ == '__main__':
    main()