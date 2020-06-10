import os
import glob
import re
from pydub import AudioSegment
import soundfile as sf
import h5py
import numpy as np
import webrtcvad

import python_speech_features
import array
import h5py_cache
import matplotlib.pyplot as plt
import wandb
import argparse
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn
from torch.nn import Linear, RNN, LSTM, GRU
import torch.nn.functional as F
from torch.nn.functional import softmax, relu
from torch.autograd import Variable
import torch.optim as optim
from sklearn import metrics
import time



class FileManager:
    def __init__(self, name, directory):
        self.name = name
        self.data = h5py.File(name + '.hdf5', 'a')

        #Setup file names
        if 'files' not in self.data:
            #Get files
            files = glob.glob(directory + '/**/*.wav', recursive=True)
            files = [f for f in files]
            #files = files[0:2]
            print(files)

            dt = h5py.special_dtype(vlen=str)
            self.data.create_dataset('files', (len(files),), dtype=dt)

            #Add file names
            for i, f in enumerate(files):
                self.data['files'][i] = f

        else:
            print('File keys already added. Skipping....')

    def get_track_count(self):
        return len(self.data['files'])

    def prepare_files(self,normalize=False,SAMPLE_RATE=16000, SAMPLE_WIDTH=2, SAMPLE_CHANNELS=1,FRAME_SIZE=480):
        '''
              Prepares the files for the project.
              Will do the following check for each file:
              1. Check if it has been converted already to the desired format.
              2. Converts all files to WAV with the desired properties.
              3. Stores the converted files in a separate folder.
        '''
        print('Found {0} tracks to check.'.format(self.get_track_count()))
        progress = 1

        #Setup raw data
        if 'raw' not in self.data:
            dt = h5py.special_dtype(vlen=np.dtype(np.int16))
            self.data.create_dataset('raw',(self.get_track_count(),), dtype=dt)
       # if 'frames_individual' not in self.data:
       #     dt = h5py.special_dtype(vlen=np.dtype(np.int16))
       #     self.data.create_dataset('frames_individual',(self.get_track_count(),), dtype=dt)


        #Convert files to desired format and save raw content
        for i, file in enumerate(self.data['files']):
            print('Processing {0} of {1}'.format(progress, self.get_track_count()),end='\r', flush=True)
            progress += 1

            #Check if file is already converted
            if len(self.data['raw'][i]) > 0:
                continue

            #Convert file to array
            track = (AudioSegment.from_file(file)
                     .set_frame_rate(SAMPLE_RATE)
                     .set_sample_width(SAMPLE_WIDTH)
                     .set_channels(SAMPLE_CHANNELS))

            #Normalize if specified
            if normalize:
                track = track.apply_gain(-track.max_dBFS)

            #Store data in raw format
            self.data['raw'][i] = np.array(track.get_array_of_samples(), dtype=np.int16)

           # frame_count = int(len(self.data['raw'][i])+(FRAME_SIZE-(len(self.data['raw'][i] % FRAME_SIZE))) / FRAME_SIZE)

           # raw = np.concatenate((self.data['raw'][i], np.zeros(FRAME_SIZE - (len(self.data['raw'][i]) % FRAME_SIZE))))
           # if len(self.data['frames_individual'][i] > 0):
           #     continue
           # self.data['frames_individual'][i] = np.array(np.split(raw, len(raw)/FRAME_SIZE))


        self.data.flush()
        print('\nDone')


    def collect_frames(self, FRAME_SIZE=480):
        '''
        Takes all the audio files and merges their frames together into one mega-large array which will be used the the sample generator
        '''
        if 'frames' in self.data:
            print('Frames are already merged. Skipping...')
            return

        if 'raw' not in self.data:
            print('Could not find raw data.')
            return


        if 'starting_point' not in self.data:
            dt = np.dtype(np.int)
            self.data.create_dataset('starting_point', (self.get_track_count(),1), dtype=dt)


        frame_count = 0
        progress = 1

        #Calculate the number of frames needed.
        self.data['starting_point'][0] = 0
        #print(self.data['starting_point'][0])
        for i, raw in enumerate(self.data['raw']):
            if i > 0:
                self.data['starting_point'][i] = frame_count - 1
            frame_count += int((len(raw) + (FRAME_SIZE - (len(raw) % FRAME_SIZE))) / FRAME_SIZE)

            print('Starting point at ', self.data['starting_point'][i])
            #print(self.data['starting_point'][i])
            #print(i)
            print('Counting frames ({0} of {1})'.format(progress, self.get_track_count()), end='\r', flush=True)
            progress += 1

        #Create dataset for frames
        dt = np.dtype(np.int16)
        self.data.create_dataset('frames', (frame_count, FRAME_SIZE), dtype=dt)

        progress = 0

        #Buffer to speed up merging
        buffer = np.array([])
        buffer_limit = FRAME_SIZE*4096

        #Merge frames
        for raw in self.data['raw']:
            #Setup raw data with zero padding on the end to fit frame size
            raw = np.concatenate((raw, np.zeros(FRAME_SIZE - (len(raw) % FRAME_SIZE))))

            #Add to buffer
            buffer = np.concatenate((buffer, raw))

            # If buffer is not filled up and we are not done, keep filling the buffer up.
            if len(buffer) < buffer_limit and progress + (len(buffer) / FRAME_SIZE) < frame_count:
                continue

            # Get frames.
            frames = np.array(np.split(buffer, len(buffer) / FRAME_SIZE))
            buffer = np.array([])

            #Add frames to list
            self.data['frames'][progress : progress + len(frames)] = frames
            progress += len(frames)
            print('Merging frames ({0} of {1})'.format(progress, frame_count), end='\r', flush=True)

        self.data.flush()
        print('Done')

    def get_track_names(self,directory):
        tracks = glob.glob(directory + '/amicorpus/*', recursive=True)
        tracks = [f[f.find('amicorpus') + 10:] for f in tracks]
        return tracks

    def get_xml_files(self, directory, track_name):
        files = glob.glob(directory+'/words/'+track_name+'.*.words.xml')
        files = [f for f in files]
        return files


    def get_timestamps(self, xml_files):
        timestamps = []
        #print('Files are,    ',xml_files)
        for i, j in enumerate(xml_files):
            tree = ET.parse(j)
            root = tree.getroot()
            start_time = end_time = 0
            for element in root:
                if element.tag == 'w' and 'starttime' in element.attrib and 'endtime' in element.attrib:
                    start_time = float(element.attrib['starttime'])
                    end_time = float(element.attrib['endtime'])
                    if end_time > start_time:
                        if start_time is not None and end_time is not None:
                            timestamps.append([start_time, end_time])
        timestamps.sort()
        #print(timestamps)
        return timestamps

    def get_timestamps_from_rttm(self,directory, name):
        rttm = open(os.getcwd()+'/rttm/'+name+'.rttm')
        lines = [line.split() for line in rttm]
        timestamps = []
        for line in lines:
            timestamps.append([int(float(line[3])/0.03),int(float(line[3])/0.03+float(line[4])/0.03)])
        rttm.close()
        return timestamps





    def label_frames(self, directory, FRAME_LENGTH):
        if 'ground_truth' not in self.data:
            frame_count = len(self.data['frames'])
            dt = np.dtype(np.uint8)
            self.data.create_dataset('ground_truth', (frame_count,),dtype=dt,data=np.zeros(frame_count))


        for i, file in enumerate(self.data['files']):
            for j, name in enumerate(self.get_track_names(directory)):
                if name in file:
                    #xml_files = self.get_xml_files(directory,track_name=name)
                    timestamps = self.get_timestamps_from_rttm(directory='rttm',name=name)

                    for x,y in enumerate(timestamps):
                        #start_frame = int(self.data['starting_point'][i]) + int(y[0]/(FRAME_LENGTH/1000))
                        #end_frame = int(self.data['starting_point'][i]) + int(y[1]/(FRAME_LENGTH/1000))
                        start_frame = int(self.data['starting_point'][i] + y[0])
                        end_frame = int(self.data['starting_point'][i] + y[1])
                        self.data['ground_truth'][start_frame:end_frame] = 1
                        #array of zeroes to contain frame labels
                        #print(self.data['raw'][i])
        self.data.flush()
        return





class Vis:
    @staticmethod
    def _norm_raw(raw):
        return raw / np.max(np.abs(raw), axis=0)

    @staticmethod
    def _time_axis(raw, labels, SAMPLE_RATE=16000):
        time = np.linspace(0, len(raw) / SAMPLE_RATE, num=len(raw))
        time_labels = np.linspace(0, len(raw)/SAMPLE_RATE, num=len(labels))
        return time, time_labels

    @staticmethod
    def _plot_waveform(frames, labels, title='Sample'):
        raw = Vis._norm_raw(frames.flatten())
        time, time_labels = Vis._time_axis(raw, labels)
        #print("labels arrrrrrrreee",labels)
        plt.figure(1, figsize=(16,3))
        plt.title(title)
        plt.plot(time, raw)
        plt.plot(time_labels, labels-0.5)
        plt.show()

    @staticmethod
    def plot_sample(frames, labels, title='Sample', show_distribution = True):
        Vis._plot_waveform(frames, labels, title)

        if show_distribution:
            voice = (labels.tolist().count(1) * 100) / len(labels)
            silence = (labels.tolist().count(0) * 100) / len(labels)
            print('{0:.0f} % voice {1:.0f} % silence'.format(voice, silence))

    @staticmethod
    def plot_sample_webrtc(frames, sensitivity=0, SAMPLE_RATE=16000):
        '''
        Plot a sample labeled with WebRTC VAD
        (after noise is applied to sample).
        Sensitivity is an integer from 0 to 2,
        with 0 being the most sensitive.
        '''
        vad = webrtcvad.Vad(sensitivity)
        labels = np.array([1 if vad.is_speech(f.tobytes(), sample_rate=SAMPLE_RATE) else 0 for f in frames])
        Vis._plot_waveform(frames, labels, title='Sample (WebRTC)')

    @staticmethod
    def plot_features(mfcc=None, delta=None):
        '''
        Plots the MFCC and delta-features
        for a given sample.
        '''
        if mfcc is not None:
            plt.figure(1, figsize=(16, 3))
            plt.plot(mfcc)
            plt.title('MFCC ({0} features)'.format(mfcc.shape[1]))
            plt.show()

        if delta is not None:
            plt.figure(1, figsize=(16, 3))
            plt.plot(delta)
            plt.title('Deltas ({0} features)'.format(mfcc.shape[1]))
            plt.show()

def extract_features(speech_frames, align_frames, SAMPLE_WIDTH = 2, SAMPLE_RATE=16000, SAMPLE_CHANNELS=1,mfcc_window_size=4, FRAME_SIZE_MS=30):
    #Get frames data from track
    frames_aligned = np.concatenate((align_frames,speech_frames))
    mfcc = python_speech_features.mfcc(frames_aligned,SAMPLE_RATE,winstep=(FRAME_SIZE_MS/1000),winlen=mfcc_window_size*(FRAME_SIZE_MS/1000), nfft=2048)
    #First MFCC feature is just the DC offset
    mfcc = mfcc[:, 1:]
    delta = python_speech_features.delta(mfcc, 2)
    return speech_frames, mfcc, delta


class DataGenerator:
    def __init__(self, data, size_limit=0):
        self.data = data
        self.size = size_limit if size_limit > 0 else len(data['labels'])
        self.data_mode = 0 #Default to training data

    def setup_generation(self, frame_count, step_size, batch_size, val_part=0.1, test_part=0.1):
        self.frame_count = frame_count
        self.step_size = step_size
        self.batch_size = batch_size

        #Setup indexes and size for data splits
        self.train_index = 0
        self.val_index = int((1.0 - val_part - test_part) * self.size)
        self.test_index = int((1.0-test_part)*self.size)

        self.train_size = self.val_index
        self.val_size = self.test_index-self.val_index
        self.test_size = self.size - self.test_index

    def use_train_data(self):
        #Calculate how many batches we can construct from our given parameters
        n = int((self.train_size - self.frame_count) / self.step_size) + 1
        self.batch_count = int(n/self.batch_size)
        self.initial_pos = self.train_index
        self.data_mode = 0

    def use_validate_date(self):
        # Calculate how many batches we can construct from our given parameters.
        n = int((self.val_size - self.frame_count) / self.step_size) + 1
        self.batch_count = int(n / self.batch_size)
        self.initial_pos = self.val_index
        self.data_mode = 1

    def use_test_data(self):
        # Calculate how many batches we can construct from our given parameters.
        n = int((self.test_size - self.frame_count) / self.step_size) + 1
        self.batch_count = int(n / self.batch_size)
        self.initial_pos = self.test_index
        #print(self.initial_pos, self.batch_count, n, self.batch_size)
        self.data_mode = 2

    def get_data(self, index_from, index_to, test=False):
        frames = self.data['frames'][index_from:index_to]
        mfcc = self.data['mfcc'][index_from:index_to]
        delta = self.data['delta'][index_from:index_to]
        labels = self.data['labels'][index_from:index_to]
        if test:
            print("TEST frames {} labels {}".format(len(frames), len(labels)))
        return frames, mfcc, delta, labels

    def get_batch(self, index):

        # Get current position.
        pos = self.initial_pos + (self.batch_size * index) * self.step_size

        # Get all data needed.
        l = self.frame_count + self.step_size * self.batch_size
        frames, mfcc, delta, labels = self.get_data(pos, pos + l)

        x, y, i = [], [], 0

        # Get batches
        while len(y) < self.batch_size:
            # Get data for the window.
            X = np.hstack((mfcc[i: i + self.frame_count], delta[i: i + self.frame_count]))

            # Append sequence to list of frames
            x.append(X)

            # Select label from center of sequence as label for that sequence.
            y_range = labels[i: i + self.frame_count]
            y.append(int(y_range[int(self.frame_count / 2)]))

            # Increment window using set step size
            i += self.step_size

        return x, y

    def plot_data(self, index_from, index_to, show_track=False):

        frames, mfcc, delta, labels = self.get_data(index_from, index_to)

        Vis.plot_sample(frames, labels)
        Vis.plot_sample_webrtc(frames)
        Vis.plot_features(mfcc, delta)

        # By returning a track and having this as the last statement in a code cell,
        # the track will appear as an audio track UI element (not supported by Windows).



def count_params(net, verbose = True):
    count = sum(p.numel() for p in net.parameters())
    if verbose:
        print(f'Model parameters: {count}')
    return count

class Net(nn.Module):
    def __init__(self, large=True, lstm=True, CUDA=True, FEATURES=24, FRAMES=30, BATCH_SIZE=2048, STEP_SIZE=6):
        super(Net, self).__init__()
        self.large = large
        self.lstm = lstm
        self.relu = nn.ReLU
        self.FEATURES = FEATURES
        self.FRAMES = FRAMES
        self.BATCH_SIZE = BATCH_SIZE
        self.STEP_SIZE = STEP_SIZE
        if CUDA:
            self.CUDA = True
        else:
            self.CUDA = False
        #Batch first = True to return output with batch in first dimension
        if lstm:
            self.hidden = self.init_hidden()
            self.rnn = LSTM(input_size=FEATURES, hidden_size=FRAMES, num_layers=1, batch_first=True)
        else:
            self.rnn = GRU(input_size=FEATURES, hidden_size=FRAMES, num_layers=1, batch_first=True)

        if large:
            self.lin1 = nn.Linear(FRAMES**2, 26)
            self.lin2 = nn.Linear(26, 2)
        else:
            self.lin = nn.Linear(FRAMES**2, 2)

        self.softmax = nn.Softmax(dim=1)

    def init_hidden(self):
        h = Variable(torch.zeros(1, self.BATCH_SIZE, self.FRAMES))
        c = Variable(torch.zeros(1, self.BATCH_SIZE, self.FRAMES))

        if self.CUDA:
            h = h.cuda() #Can also use .to(device)
            c = c.cuda()

        return h,c

    def forward(self,x):
        # (batch, frames, features)
        if hasattr(self,'lstm') and self.lstm:
            x, _ = self.rnn(x, self.hidden)
        else:
            x, _ = self.rnn(x)
        x = x.contiguous().view(-1, self.FRAMES **2) #See https://stackoverflow.com/questions/48915810/pytorch-contiguous for more info on why we use .contiguous()

        #(batch, units)
        if self.large:
            #x = self.lin1(x)
            x = F.relu(self.lin1(x))
            #x = self.relu(x)
            x = self.lin2(x)
        else:
            x = self.lin(x)

        return self.softmax(x)


def accuracy(out, y):
    '''
    Calculate accuracy of model
     out.shape ==> (64,2), y.shape ==> (64)
    '''
    out = torch.max(out, 1)[1].float()
    eq = torch.eq(out, y.float()).float() #eq computes element-wise equality
    return torch.mean(eq)


def test_net(net, num_batches, generator):
    '''
    Test function to test given network with a view batches
    :param net: the network
    :param num_batches: how many batches you want to test for
    :param generator: data generator object
    :return: (network output, network accuracy)
    '''
    for i in range(num_batches):
        #Get batch
        X, y  = generator.get_batch(i)
        X = torch.from_numpy(np.array(X)).float().cpu()
        y = torch.from_numpy(np.array(y)).cpu()


        #Run through the network, use .data.numpy() to convert tensor to numpy array
        out = net(X)
        acc = accuracy(out, y).data.numpy()
        print('Example output: ', out.data.numpy()[0])
        #print('Output size: ', out.data.numpy().size())
        print('Example accuracy: ', acc)

def net_path(epoch, title, no_epoch=True):
    path = os.getcwd()+'/models/'+title
    if epoch >= 0:
        return path + '_epoch'+str(epoch).zfill(3)+'.net'
    elif no_epoch:
        return path + '.net'

def save_net(net, epoch, title='net'):
    if not os.path.exists(os.getcwd()+'/models'):
        os.makedirs(os.getcwd()+'/models')
    torch.save(net, net_path(epoch,title,no_epoch=True))

def load_net(epoch=14, title='net', CUDA=True):
    if CUDA:
        return torch.load(net_path(epoch, title, True))
    else:
        return torch.load(net_path(epoch, title, True), map_location='cpu')

def train_net(net, data, size_limit=0, epochs=30, lr=1e-3, use_adam=True,
              weight_decay=1e-5, momentum=0.9,
              early_stopping=False, patience=25, frame_count=30, step_size=6,
              auto_save=True,batch_size=2048, title='net', verbose=True, USE_CUDA=True):
    '''
    Training of neural network
    :param net: Your network
    :param data: Your data (.hdf5 object)
    :param size_limit: size limit for data generator
    :param epochs: # of epochs to train for
    :param lr: learning rate
    :param use_adam: the other option is to use SGD
    :param weight_decay: use weight decay (for ADAM)
    :param momentum: momentum to use with ADAM/SGD
    :param early_stopping: early stopping, if it is true also use patience parameter. Also make sure that patience parameter is smaller than number of epochs (duh)
    :param patience:  # of epochs early stopping should wait before stopping training
    :param frame_count: # of frames given to network at a time
    :param step_size: # of steps taken (e.g. for step size of 6 the first input will be frame 1-30, second input will be frame 7-36
    :param auto_save: save net after each epoch?
    :param title: save net as ...
    :param verbose: plot stats after each epoch?
    :return:
    '''
    #Set up instance of data generator using default partitions


    generator = DataGenerator(data, size_limit)
    generator.setup_generation(frame_count, step_size, batch_size)

    if generator.train_size == 0:
        print('Training data not found')
        return

    #Use CUDA?
    if USE_CUDA:
        net.cuda()

    #loss function
    criterion = nn.CrossEntropyLoss ()
    if USE_CUDA:
        criterion.cuda()

    #Instantiate optimizer
    if use_adam:
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    #If verbose, print staring conditions
    print('Initiating training of {}... \n\nLearning Rate {}'.format(title,lr))

    net.train()
    stalecount, maxacc = 0, 0

    def plot(losses, accs, val_losses, val_accs):
        '''
        Continuously plots the training/validation loss and accuracy of
        the model being trained. This function is only called if verbose is True
        for training session
        '''
        e = [i for i in range(len(losses))]
        fig = plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        plt.plot(e, losses, label='Loss (Validation)')

        if generator.val_size != 0:
            plt.plot(e,val_losses, label='Loss (Validation)')

        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(e, accs, label='Accuracy (Training)')

        if generator.val_size != 0:
            plt.plot(e, val_accs, label='Accuracy (Validation)')

        plt.legend()
        plt.show()

    def run(net, optimize=False):
        '''
        This function constitutes a single epoch.
        Snippets are loaded into memory and associated frames are loaded as generators.
        As new frames are needed they are generated by the iterator, and not stored in the memory when not used.
        If optimize is True, associated optimizer will backprop and adjust network weights. Returns average sample loss
        and accuracy for that epoch.
        '''
        epoch_loss, epoch_acc = 0, []

        batches = generator.batch_count
        if batches == 0:
            raise ValueError('Not enough data to create a full batch.')

        #Helper function responsible for running a batch
        def run_batch(X, y, epoch_loss, epoch_acc):
            X = Variable(torch.from_numpy(np.array(X)).float())
            y = Variable(torch.from_numpy(np.array(y))).long()

            if USE_CUDA:
                X = X.cuda()
                y = y.cuda()

            out = net(X)

            #Compute loss and accuracy for batch
            batch_loss = criterion(out, y)
            batch_acc = accuracy(out, y)

            #If training session, initiate backpropogation and optimization
            if optimize == True:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            if USE_CUDA:
                batch_acc = batch_acc.cpu()
                batch_loss = batch_loss.cpu()

            #Accumulate loss and accuracy for epoch metrics
            epoch_loss += batch_loss.data.numpy() / float(batch_size)
            epoch_acc.append(batch_acc.data.numpy())
            #print(batch_loss, batch_acc)


            return epoch_loss, epoch_acc

        #run network for each batch
        for i in range(batches):
            #Get a new batch and run it
            X,y = generator.get_batch(i)
            temp_loss, temp_acc = run_batch(X, y, epoch_loss, epoch_acc)
            epoch_loss += temp_loss / float(batches)

        print('{} {}'.format(epoch_loss, np.mean(temp_acc)))


        return epoch_loss, np.mean(temp_acc)

    losses, accs, val_losses, val_accs = [], [], [], []
    if verbose:
        start_time = time.time()
    #Iterate over training epochs
    for epoch in range(epochs):
        #Calculate loss and accuracy for that epoch and optimize
        generator.use_train_data()
        loss, acc = run(net, optimize=True)
        losses.append(loss)
        accs.append(acc)

        #If validation data is available, calculate validation metrics
        if generator.val_size != 0:
            net.eval()
            generator.use_validate_date()
            val_loss, val_acc = run(net)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            net.train()
            print("Epoch {} Loss {} Accuracy {}".format(epoch,val_loss,val_acc))

            #Early stopping
            #If validation accuracy does not improve for
            #a set amount of epochs, stop training
            #save best model

            if epoch > 0 and val_accs[-1] <= maxacc:
                stalecount += 1
                if stalecount > patience and early_stopping:
                    return
            else:
                stalecount = 0
                maxacc = val_accs[-1]

        if auto_save:
            save_net(net, epoch, title)

        if verbose:
            if epoch == 0:
                dur = str(int((time.time() - start_time) / 60))
                print(f'\nEpoch wall-time: {dur} min')

            plot(losses, accs, val_losses, val_accs)

def set_seed(seed=1500, CUDA=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if CUDA:
        torch.cuda.manual_seed_all(seed)

def test_predict(net, data, size_limit, noise_levels,FRAMES = 30,STEP_SIZE = 6,BATCH_SIZE=2048, CUDA=True):
    '''
    Runs given network on test data
    '''
    #Set up instance of data generator
    generator = DataGenerator(data, size_limit)
    generator.setup_generation(FRAMES, STEP_SIZE, BATCH_SIZE)
    if generator.test_size == 0:
        print('No test data found')
        return

    net.eval()
    generator.use_test_data()

    y_true, y_score = [], []

    for i in range(generator.batch_count):
        X, y = generator.get_batch(i)
        X = Variable(torch.from_numpy(np.array(X)).float())
        y = Variable(torch.from_numpy(np.array(y)).long())

        if CUDA:
            X = X.cuda()

        out = net(X)

        if CUDA:
            out = out.cpu()
            y = y.cpu()

        #Add true labels
        y_true.extend(y.data.numpy())

        #Add probabilities for positivie labels
        y_score.extend(out.data.numpy()[: 1])

    return y_true, y_score

class BiRNN(nn.Module):
    '''
    Bi-directional layer of gated recurrent units.
    Includes a fully connected layer to produce binary output.
    '''

    def __init__(self, num_in, num_hidden, batch_size=2048, large=True, lstm=False, fcl=True, bidir=False,OBJ_CUDA=True):
        super(BiRNN, self).__init__()

        self.num_hidden, self.batch_size = num_hidden, batch_size
        self.lstm, self.bidir, self.layers = lstm, bidir, 2 if large else 1
        if OBJ_CUDA:
            self.CUDA = True
        if lstm:
            self.hidden = self.init_hidden()
            self.rnn = LSTM(num_in, num_hidden, num_layers=self.layers, bidirectional=self.bidir, batch_first=True)
            sz = 18 if large else 16
        else:
            self.rnn = GRU(num_in, num_hidden, num_layers=self.layers, bidirectional=self.bidir, batch_first=True)
            sz = 18

        embed_sz = num_hidden * 2 if self.bidir or self.layers > 1 else num_hidden

        if not fcl:
            self.embed = nn.Linear(embed_sz, 2)
        else:
            if large:
                self.embed = nn.Sequential(
                    nn.Linear(embed_sz, sz + 14),
                    nn.BatchNorm1d(sz + 14),
                    nn.Dropout(p=0.2),
                    nn.ReLU(),
                    nn.Linear(sz + 14, sz),
                    nn.BatchNorm1d(sz),
                    nn.Dropout(p=0.2),
                    nn.ReLU(),
                    nn.Linear(sz, 2)
                )
            else:
                self.embed = nn.Sequential(
                    nn.Linear(embed_sz, sz),
                    nn.BatchNorm1d(sz),
                    nn.Dropout(p=0.2),
                    nn.ReLU(),
                    nn.Linear(sz, 2)
                )

    def init_hidden(self):
        num_dir = 2 if self.bidir or self.layers > 1 else 1
        h = Variable(torch.zeros(num_dir, self.batch_size, self.num_hidden))
        c = Variable(torch.zeros(num_dir, self.batch_size, self.num_hidden))

        if self.CUDA:
            h = h.cuda()
            c = c.cuda()

        return h, c

    def forward(self, x):
        if self.CUDA:
            self.rnn.flatten_parameters()

        x = x.permute(0, 2, 1)

        if self.lstm:
            x, self.hidden = self.rnn(x, self.hidden)
        else:
            x, self.hidden = self.rnn(x)

        # Extract outputs from forward and backward sequence and concatenate
        # If not bidirectional, only use last output from forward sequence
        x = self.hidden.view(self.batch_size, -1)

        # (batch, features)
        return self.embed(x)


def netvad(net, data, noise_level='-3', init_pos=700, length=100000, only_plot_net=False, timeit=True,FRAMES=30, STEP_SIZE=6, BATCH_SIZE=2048, FEATURES=24,OBJ_CUDA=True):
    '''
    Generates a sample of specified length and runs it through
    the given network. By default, the network output is plotted
    alongside the original labels and WebRTC output for comparison.
    '''

    # Set up an instance of data generator using default partitions
    generator = DataGenerator(data)
    generator.setup_generation(FRAMES, STEP_SIZE, BATCH_SIZE)

    if generator.test_size == 0:
        print('Error: no test data was found!')
        return

    net.eval()
    generator.use_test_data()
    #print("Data {}".format(generator.test_size))

    i = generator.initial_pos
    raw_frames, mfcc, delta, labels = generator.get_data(init_pos, init_pos + length, test=True)





    # Convert sample to list of frames
    def get_frames():
        i = 0
        while i < length - FRAMES:
            yield np.hstack((mfcc[i: i + FRAMES], delta[i: i + FRAMES]))
            i += 1

    # Start timer
    if timeit:
        start_net = time.time()

    # Creates batches from frames
    frames = list(get_frames())
    batches, i, num_frames = [], 0, -1
    while i < len(frames):
        full = i + BATCH_SIZE >= len(frames)
        end = i + BATCH_SIZE if not full else len(frames)
        window = frames[i:end]
        if full:
            num_frames = len(window)
            while len(window) < BATCH_SIZE:
                window.append(np.zeros((FRAMES, FEATURES)))
        batches.append(np.stack(window))
        i += BATCH_SIZE

    # Predict for each frame
    offset = 15
    accum_out = [0] * offset
    for batch in batches:
        X = Variable(torch.from_numpy(batch).float())
        X = X.cuda()
        if OBJ_CUDA:
            out = torch.max(net(X), 1)[1].cpu().float().data.numpy()
            print(out)
        else:
            out = torch.max(net(X), 1)[1].float().data.numpy()
        accum_out.extend(out)

    # Stop timer
    if timeit:
        dur_net = str((time.time() - start_net) * 1000).split('.')[0]
        device = 'GPU' if OBJ_CUDA else 'CPU'
        seq_dur = int((length / 100) * 3)
        print(f'Network processed {len(batches) * BATCH_SIZE} frames ({seq_dur}s) in {dur_net}ms on {device}.')

    # Adjust padding
    if num_frames > 0:
        accum_out = accum_out[:len(accum_out) - (BATCH_SIZE - num_frames)]
    accum_out = np.array(accum_out)

    frames = np.array(frames)

    # Cut frames outside of prediction boundary
    raw_frames = raw_frames[offset:-offset]
    labels = labels[offset:-offset]
    accum_out = accum_out[offset:]

    # Plot results
    #print('Displaying results for noise level:', noise_level)
    #if not only_plot_net:
    #    Vis.plot_sample(raw_frames, labels, show_distribution=False)
    #    Vis.plot_sample_webrtc(raw_frames, sensitivity=0)
    print(accum_out)
    Vis.plot_sample(raw_frames, labels, title='Sample (Neural Net)', show_distribution=False)





def main():
    '''
    Set parameters
    '''
    parser = argparse.ArgumentParser(description='VAD on the AMI corpus')
    parser.add_argument('--SRate', type=int, default=16000, metavar='SR',
                        help="Sample rate to be used")
    parser.add_argument('--SChannels', type=int, default=1, metavar='SC',
                        help="Number of channels")
    parser.add_argument('--SWidth', type=int, default=2, metavar='SW',
                        help="Sample Width")
    parser.add_argument('--FSize', type=int, default=30, metavar='FS',
                        help="Frame size in MS")
    parser.add_argument('--Path', type=str, default='/home/lucas/PycharmProjects/Papers_with_code/data/AMI', metavar='DP',
                        help="Path to data folder")
    parser.add_argument('--MFCC_WinS', type=int, default=4, metavar='MWS',
                        help="MFCC window size (in frames)")
    parser.add_argument('--Generate_Datasets', type=bool, default=True, metavar='GD',
                        help="Generate hdf5 datasets? (Default=True)")
    parser.add_argument('--Batch_Size', type=int, default=2048, metavar='BS',
                        help="Batch size (default 2048)")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()

    #Calculate frame size in data points
    FRAME_SIZE = int(args.SRate * (args.FSize / 1000.0))

    #Change path
    PATH = args.Path
    os.chdir(PATH)

    AMI_speech = FileManager('AMI_speech_test', 'amicorpus')
    #AMI_speech.prepare_files(normalize=False, SAMPLE_RATE=args.SRate, SAMPLE_WIDTH=args.SWidth,
     #                        SAMPLE_CHANNELS=args.SChannels)
    #AMI_speech.collect_frames(FRAME_SIZE=FRAME_SIZE)
    #AMI_speech.label_frames(PATH,FRAME_LENGTH=30)
    speech_data = AMI_speech.data
    data = h5py_cache.File('data_AMI_test.hdf5', 'a', chunk_cache_mem_size=1024 ** 3)

    CUDA = True

    SLICE_MIN_MS = 2500
    SLICE_MAX_MS = 5000

    # Frame size to use for the labelling.
    #FRAME_SIZE_MS = 30

    # Convert slice ms to frame size.
    SLICE_MIN = int(SLICE_MIN_MS / 30)
    SLICE_MAX = int(SLICE_MAX_MS / 30)

    if 'labels' not in data:
        print('Preparing Data...')
        pos = 0
        l = len(AMI_speech.data['frames'])

        slices = []

        # Split frames into slices for feature extraction. Slices are used to extract features is "batches" to not overload system memory
        while pos + SLICE_MIN < l:
            slice_indexing = (pos, pos + SLICE_MIN)
            slices.append(slice_indexing)
            pos = slice_indexing[1]

        # Add remainder to last slice.
        slices[-1] = (slices[-1][0], l)

        #Get total frame count
        total = l + args.MFCC_WinS

        #Create datasets
        data.create_dataset('frames', (total, FRAME_SIZE), dtype=np.dtype(np.int16))
        data.create_dataset('mfcc', (total, 12), dtype=np.dtype(np.float32))
        data.create_dataset('delta', (total, 12), dtype=np.dtype(np.float32))

        #Create data set for labels
        dt = np.dtype(np.int8)
        data.create_dataset('labels', (total,), dtype=dt)

        pos = 0

        for s in slices:
            frames = AMI_speech.data['frames'][s[0]:s[1]]
            labels = AMI_speech.data['ground_truth'][s[0]:s[1]]
            if pos == 0:
                align_frames = np.zeros((args.MFCC_WinS - 1, FRAME_SIZE))
            else:
                align_frames = data['frames'][pos-args.MFCC_WinS+1:pos]
            frames, mfcc, delta = extract_features(align_frames=align_frames,speech_frames=frames, SAMPLE_WIDTH=args.SWidth, SAMPLE_RATE=args.SRate, SAMPLE_CHANNELS=args.SChannels, mfcc_window_size=args.MFCC_WinS, FRAME_SIZE_MS=args.FSize)
            data['frames'][pos:pos + len(labels)] = frames
            data['mfcc'][pos:pos + len(labels)] = mfcc
            data['delta'][pos:pos + len(labels)] = delta

            data['labels'][pos: pos + len(labels)] = labels
            pos += len(labels)
            print('Preparing ({0:.2f} %)'.format((pos * 100) / total), end='\r', flush=True)
        data.flush()
        print('\nDone')




    # Test generator features.
    #generator = DataGenerator(data, size_limit=10000)

    #generator.setup_generation(frame_count=3, step_size=5, batch_size=4)

    #generator.use_train_data()
    #X, y = generator.get_batch(50)

    #print(f'Load a few frames into memory:\n{X[0]}\n\nCorresponding label: {y[0]}')
    #print(len(AMI_speech.data['frames']))

    #Vis.plot_sample(frames=AMI_speech.data['frames'][1:],labels=AMI_speech.data['ground_truth'][1:])
    #Vis.plot_sample_webrtc(frames=AMI_speech.data['frames'][1:10000], sensitivity=0)
    #generator.plot_data(0, 400000)
    #print(data['frames'][0:2])

    #net = Net(large=False)
    #count_params(net)
    #print(net)

    # Test generator
    generator = DataGenerator(data)
    generator.setup_generation(frame_count=30, step_size=6, batch_size=2048)
    generator.use_train_data()

    print(generator.batch_count, 'training batches were found')

    # Compact instantiation of untrained network on CPU
    temp, CUDA = CUDA, False
    net, CUDA = Net(large=False, CUDA=CUDA), temp
    del temp

    test_net(net=net, num_batches=1, generator=generator)


    #Test simple lstm network
    set_seed(1001,CUDA=CUDA)
    #net = BiRNN(num_in=30,num_hidden=30,batch_size=2048,large=True,lstm=False,fcl=True,bidir=True,OBJ_CUDA=True)
    net = Net(large=True,lstm=False)
    train_net(net, data=data)
    #get_xml(PATH)
    #net, data, noise_level = '-3', init_pos = 50, length = 700, only_plot_net = False, timeit = True, FRAMES = 30, STEP_SIZE = 6, BATCH_SIZE = 2048, FEATURES = 24, OBJ_CUDA = True
    netvad(net = net, data=data,init_pos=10000,length= 20000, only_plot_net=True, timeit=True, FRAMES=30, STEP_SIZE=6, BATCH_SIZE=2048, FEATURES=24, OBJ_CUDA=True)
   # print('__________________')
   # print(np.count_nonzero(data['labels']))
   # print(len(data['labels']))
    #print('__________________')
    #print(np.count_nonzero(AMI_speech.data['ground_truth']))
    #AMI_speech.get_timestamps_from_rttm(directory='rttm',name='IS1006d')
   # print(len(AMI_speech.data['ground_truth']))
   # print(np.count_nonzero(speech_data['ground_truth']))
   # print(len(speech_data['ground_truth']))
    #data['labels'] = AMI_speech.data['ground_truth']
if __name__ == '__main__':
    main()