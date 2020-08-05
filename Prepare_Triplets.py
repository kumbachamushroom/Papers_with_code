import os
import glob
import re

import h5py
import numpy as np



import array
import h5py_cache
import matplotlib.pyplot as plt
import wandb
import argparse


import torch
import torch.nn as nn
from torch.nn import Linear, RNN, LSTM, GRU
import torch.nn.functional as F
from torch.nn.functional import softmax, relu
from torch.autograd import Variable
import torch.optim as optim
from sklearn import metrics
import time
from pydub import AudioSegment
import io
import webrtcvad
import noisereduce as nr
import scipy.io.wavfile as wavfile
import struct
from scipy import signal
from scipy.fft import fftshift
import torchaudio
import torch


PATH = "/home/lucas/PycharmProjects/Papers_with_code/data/AMI"
os.chdir(PATH)
print("directory is set to ",os.getcwd())


'''Triplet generator '''
class Generate_Triplet:
    def __init__(self,filenames,path):
        self.filenames = filenames
        self.path = path
        os.chdir(self.path)

    def split_wav_files(self):
        for filename in self.filenames:
            #speaker_names = self.get_speakers(filename=filename)
            self.extract_speech(filename)


    def get_speakers(self,filename):
        os.chdir(os.path.join(self.path,'rttm'))
        print(os.getcwd())
        #rttm_files = glob.glob(pathname=filename+'.rttm',recursive=True)

        try:
            rttm = open(filename+'.rttm')
        except:
            print("RTTM file {}.rttm not found".format(filename))
        else:
            lines = [line.split() for line in rttm]
            #speakers = [line[7] for line in lines]
            speakers = [np.asarray(line[7]) for line in lines]
            speakers = np.unique(speakers)
            Headset_num = filename[-1:]
            print(Headset_num)
        #return speakers

    def get_timestamps(self,speaker_name,filename,return_path):
        os.chdir(os.path.join(self.path,'rttm'))
        rttm = open(filename+'.rttm')
        timestamps = [[int(float(line.split()[3])*16000),int(float(line.split()[4]))*16000] for line in rttm if line.split()[7] == speaker_name]
        os.chdir(return_path)
        return timestamps
        #timestamps = [np.asarray(int(int(line[])))]

    def extract_speech(self,filename):
        speaker_dict = {'Headset-0':'Speaker_A', 'Headset-1':'Speaker_B', 'Headset-2':'Speaker_C',
                        'Headset-3':'Speaker_D',  'Headset-4':'Speaker_E', 'Headset-5':'Speaker_F'}
        os.chdir(os.path.join(self.path,'amicorpus_individual/'+filename+'/audio'))
        wav_filename = glob.glob('*.wav')
        print(wav_filename)
        print(filename)


        for file in wav_filename:

            speaker = speaker_dict[file[file.index('.') + 1:file.rindex('.')]]
            timestamps = self.get_timestamps(filename=filename,speaker_name=speaker,return_path=os.getcwd())
            wav_file = (AudioSegment.from_file(str(file)).set_frame_rate(16000).set_sample_width(2).set_channels(1))
            track = np.array(wav_file.get_array_of_samples(),dtype=np.int16)
            new_wav_file = np.empty_like(track)

            for start_sample,duration_sample in timestamps:
            #count = count + duration_sample
                new_wav_file[start_sample:start_sample+duration_sample] = track[start_sample:start_sample+duration_sample]
                #new_wav_file[start_sample:start_sample+16000]=0
                #new_wav_file[start_sample+duration_sample-16000]=0
            cwd = os.getcwd()
            os.chdir(os.path.join(self.path,'triplets'))
            new_wav_file = new_wav_file[new_wav_file != 0]
            s = io.BytesIO(new_wav_file)
            wav = AudioSegment.from_raw(s,sample_width=2,frame_rate=16000,channels=1).export(filename+'_'+speaker+'.wav',format='wav')
            os.chdir(cwd)


    def voice_activity(self,filename):
        os.chdir(os.path.join(self.path,'amicorpus_individual',filename,'audio'))
        files = glob.glob('*.wav',recursive=True)
        file = files[3]
        print(file)
        sample_rate, samples = wavfile.read(file)
        vad = webrtcvad.Vad()
        vad.set_mode(3)
        raw_samples = struct.pack("%dh" % len(samples), *samples)
        window_duration = 0.03
        samples_per_window = int(window_duration*sample_rate+0.5)
        bytes_per_sample = 2

        segments= []
        count = 0
        for start in np.arange(0, len(samples), samples_per_window):
            stop = min(start + samples_per_window, len(samples))
            is_speech = vad.is_speech(raw_samples[start*bytes_per_sample:stop*bytes_per_sample], sample_rate=16000)
            segments.append(dict(start = start, stop = stop, is_speech= is_speech))

        start = 0
        end = False
        end_index = 0
        refined_segments = []

        for i in range(len(segments)):
            if segments[i]['is_speech']:
                if count == 0:
                    start = i
                count += 1

                end = False
            else:
                if count > 0:
                    end_index = i
                    end = True

            if end:
                if count < 16:
                    try:
                        while start <= end_index:
                            segments[start]['is_speech'] = False
                            start += 1
                    except:
                        print("Some fucking error {} {}".format(start,end_index))
                count = 0
                end = False


        #print(len(segments['is_speech']))
        speech_samples = np.concatenate([ samples[segment['start']:segment['stop']] for segment in segments if segment['is_speech']])

        print(speech_samples)

        #segments = []
        #for start in np.arange(0, len(samples), samples_per_window):
        #    stop = min(start + samples_per_window, len(samples))
        #    is_speech = vad.is_speech(raw_samples[start * bytes_per_sample:stop * bytes_per_sample], sample_rate=16000)

        #    segments.append(dict(start=start, stop=stop, is_speech=is_speech))

        #speech_samples = np.concatenate(
        #    [samples[segment['start']:segment['stop']] for segment in segments if segment['is_speech']])
        #print(speech_samples)
        os.chdir(os.path.join(self.path,'triplets'))
        s = io.BytesIO(speech_samples)
        wav = AudioSegment.from_raw(s, sample_width=2, frame_rate=16000, channels=1).export(
            'webrtcinditest0_and_counter.wav', format='wav')

    def Generate_Spectrogram(self,filename):
        os.chdir(os.path.join(self.path, 'triplets'))
        wav_files = glob.glob(filename+'*.wav')[0]

        #sample_rate, samples = wavfile.read(PATH+'/amicorpus/EN2001a/audio/EN2001a.Mix-Headset.wav')
        sample_rate, samples = wavfile.read(wav_files)
        frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate,mode='magnitude')
        plt.pcolormesh(times, frequencies, np.log(spectrogram))
        #plt.imshow(spectrogram)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()

        wav_files = glob.glob(filename + '*.wav')[1]

        # sample_rate, samples = wavfile.read(PATH+'/amicorpus/EN2001a/audio/EN2001a.Mix-Headset.wav')
        sample_rate, samples = wavfile.read(wav_files)
        frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate,mode='magnitude')
        plt.pcolormesh(times, frequencies, np.log(spectrogram))
        #plt.imshow(spectrogram)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()

        wav_files = glob.glob(filename + '*.wav')[2]

        # sample_rate, samples = wavfile.read(PATH+'/amicorpus/EN2001a/audio/EN2001a.Mix-Headset.wav')
        sample_rate, samples = wavfile.read(wav_files)
        frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate,mode='magnitude')
        plt.pcolormesh(times, frequencies, np.log(spectrogram))
        # plt.imshow(spectrogram)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()

        wav_files = glob.glob(filename + '*.wav')[3]

        # sample_rate, samples = wavfile.read(PATH+'/amicorpus/EN2001a/audio/EN2001a.Mix-Headset.wav')
        sample_rate, samples = wavfile.read(wav_files)
        frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate,mode='magnitude')
        plt.pcolormesh(times, frequencies, np.log(spectrogram))
        # plt.imshow(spectrogram)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()

    def Generate_Spectrogram_PyAudio(self, filename):
        #Get array of samples
        os.chdir(os.path.join(self.path,'triplets'))
        wav_files = glob.glob('*.wav')[0]
        waveform, sample_rate = torchaudio.load(wav_files)
        print(waveform.size())
        print(waveform[0,0:5])
        print(sample_rate)
        #Show first 5 seconds of waveform
        plt.figure()
        #plt.plot(waveform[0:80000].t().numpy())
        plt.show()
        #Get spectrogram
        specgram = torchaudio.transforms.Spectrogram(normalized=True, power=1,n_fft=400,hop_length=100)(waveform[:,0:48000])
        print(specgram)

        print("Shape of spectorgram: {}".format(specgram.size()))
        plt.figure()
        plt.imshow(specgram.log2()[0,:,:].numpy(), cmap='magma')
        plt.show()

        wav_files = glob.glob('*.wav')[1]
        waveform, sample_rate = torchaudio.load(wav_files)
        print(waveform.size())
        print(waveform[0, 0:5])
        print(sample_rate)
        # Show first 5 seconds of waveform
        plt.figure()
        # plt.plot(waveform[0:80000].t().numpy())
        plt.show()
        # Get spectrogram
        specgram = torchaudio.transforms.Spectrogram(normalized=True, power=1, n_fft=400, hop_length=100)(
            waveform[:, 0:48000])
        print(specgram)

        print("Shape of spectorgram: {}".format(specgram.size()))
        plt.figure()
        plt.imshow(specgram.log2()[0, :, :].numpy(), cmap='magma')
        plt.show()

    def split_speaker(self, filename, split_length_seconds):
        os.chdir(os.path.join(self.path,'triplets'))
        #sampling_rate = 16
        #speaker_file = (AudioSegment.from_file(str(filename)).set_frame_rate(16000).set_sample_width(2).set_channels(1))
        #speaker_file = np.array(speaker_file.get_array_of_samples(), dtype=np.int16)
        sampling_rate, speaker_file = wavfile.read(filename)
        speaker_file = speaker_file[0:len(speaker_file)-(len(speaker_file) % (split_length_seconds*sampling_rate))]
        #speaker_file = np.concatenate((speaker_file, np.zeros((split_length_seconds*sampling_rate)-len(speaker_file) % (split_length_seconds*sampling_rate))))
        speaker_file= np.array(np.split(speaker_file,(len(speaker_file)/(split_length_seconds*sampling_rate)),axis=0))
        #Save the audio snippets
        #print(len(speaker_file))
        os.chdir(os.path.join(self.path,'triplet_splits'))
        #wavfile.write(filename='sanitycheck.wav',rate=sampling_rate,data=speaker_file)
        for i in range(len(speaker_file)):
            file = filename[0:filename.index('.')]+'_'+str(i)+'.wav'
            wavfile.write(file, sampling_rate,speaker_file[i])

    def generate_sample_list(self,filenames,max_samples=200):
        count = 0
        os.chdir(os.path.join(self.path,'triplet_splits'))
        anchor = []
        positive = []
        samples = []

        label = 1
        for file in filenames:
            snippets = glob.glob(file+'_*.wav',recursive=True)
            #print(len(snippets))
            try:
                snippets = snippets[0:max_samples]
            except:
                continue
            #samples.append([snippets[0], label])
            for i in range(len(snippets)-1):
                for j in range(i+1, len(snippets)):
                        anchor.append([snippets[i],label])
                        positive.append([snippets[j],label])
                samples.append([snippets[i],label])
            samples.append([snippets[-1],label])
            label += 1

        f = open('anchor_pairs.txt','w+')
        for i in range(len(anchor)):
            f.write(str(anchor[i][0])+"\t"+str(positive[i][0])+"\t"+str(positive[i][1])+"\n")
        f.close()

        f = open('sample_list.txt','w+')
        for i in range(len(samples)):
            f.write(str(samples[i][0])+"\t"+str(samples[i][1])+"\n")
        f.close()






            #for i in range(int((len(snippets)*(len(snippets)-1))/2)-1):
              #  for j in range(i+1,int((len(snippets)*(len(snippets)-1))/2),)



TripletClass = Generate_Triplet(['EN2001a','EN2001b','ES2002c','ES2002d','ES2003c','ES2003d'],PATH)

#TripletClass.split_wav_files()
#TripletClass.Generate_Spectrogram_PyAudio(filename='EN2001a')
#TripletClass.split_speaker('ES2002a_Speaker_D.wav',split_length_seconds=3)
TripletClass.generate_sample_list(filenames = ['EN2001a_Speaker_A','EN2001a_Speaker_B','EN2001a_Speaker_C','EN2001a_Speaker_D','EN2001a_Speaker_E'],max_samples=150)





