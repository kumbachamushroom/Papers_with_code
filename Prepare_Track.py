import os
import glob
import torch
import torchaudio
from math import floor
import numpy as np
import pandas as pd

class Prepare_Track:

    def __init__(self, track_name, rttm_name, path_to_track, path_to_rttm):
        self.path = path_to_track
        self.path_rttm = path_to_rttm
        self.filename = track_name
        self.rttm_name = rttm_name

    def get_frames(self,window_size, step_size):
        track, sample_rate = torchaudio.load(os.path.join(self.path, self.filename+'.wav')) #load track
        track_length = len(track.numpy()[0]) #Get number of samples (length) of track
        del track #We only need the number of samples (length) of the track
        n_increments = floor(((floor(track_length/sample_rate)-window_size)/step_size)) #Number of steps taken by sliding window
        frame_list = []
        for i in range(n_increments+2): #create list of size N_frames that contain the start and stop time of each frame (convert time to samples)
            start_time = i*step_size
            end_time = start_time + window_size
            frame_list.append((start_time, end_time))
        return frame_list
        #print(frame_list)

    def create_dictionary(self, labels):
        #Create a dictionary of labels which will correspond to the label array, to be used as reference
        dict = {}
        for i in range(len(labels)):
            if labels[i] not in dict.keys():
                dict[labels[i]] = i
        return dict


    def label_frames(self, window_size, step_size):
        try:
            rttm = open(os.path.join(self.path_rttm,self.rttm_name))
        except:
            print('RTTM NOT FOUND')
        else:
            lines = [line.split() for line in rttm if line.split()[1] == self.filename]
            speakers = [np.asarray(line[7]) for line in lines]
            speakers = np.unique(speakers) #array of speaker names
            frame_list = self.get_frames(window_size=window_size, step_size=step_size)
            speaker_dict = self.create_dictionary(speakers)
            label_array = np.zeros(len(frame_list)*len(speaker_dict.keys())).reshape(len(speaker_dict.keys()),len(frame_list)) #create array of labels of size (speakers, frames)
            print(speaker_dict)
            for i, time in enumerate(frame_list):
                start_time, stop_time = time[0], time[1]
                utterances = [line for line in lines if (start_time <= float(line[3]) <= stop_time)]# and ((stop_time-float(line[3]))>=(stop_time/6))]
                for k, line in enumerate(utterances):
                    label_array[list(speaker_dict).index(line[7]),i] = 1

                print(" Labeling frame {}/{} {} ".format(i,len(frame_list), len(lines)),end='\r', flush=True)

            print("Frame labelling succesfull")
            label_df = pd.DataFrame(data=label_array,index=speaker_dict.keys())

            #label_array = np.ones(len(frame_list) * len(speaker_dict.keys())).reshape(len(speaker_dict.keys()),
                                                                                       #len(frame_list))
            #label_df_test = pd.DataFrame(data=label_array,index=speaker_dict.keys())
            #df_inner = label_df_test*label_df
            #print(label_df_test)
            return label_df, frame_list, speaker_dict






#test_obj = Prepare_Track(track_name='EN2001a.Mix-Headset',rttm_name='MixHeadset.train.rttm',path_to_track='/home/lucas/PycharmProjects/Papers_with_code/data/AMI/pyannote/amicorpus/EN2001a/audio', path_to_rttm='/home/lucas/PycharmProjects/Papers_with_code/data/AMI/pyannote/AMI')
#test_obj.label_frames(window_size=3, step_size=0.1)