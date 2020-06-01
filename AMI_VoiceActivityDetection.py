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


class FileManager:
    def __init__(self, name, directory):
        self.name = name
        self.data = h5py.File(name + '.hdf5', 'a')

        #Setup file names
        if 'files' not in self.data:
            #Get files
            files = glob.glob(directory + '/**/*.wav', recursive=True)
            files = [f for f in files]

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
            frame_count += int((len(raw) + (FRAME_SIZE - (len(raw) % FRAME_SIZE))) / FRAME_SIZE)
            self.data['starting_point'][i] = frame_count-1
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
        files = glob.glob(directory+'/segments/'+track_name+'.*.segments.xml')
        files = [f for f in files]
        return files

    def get_timestamps(self, xml_files):
        timestamps = []
        for i, j in enumerate(xml_files):
            tree = ET.parse(j)
            root = tree.getroot()
            start_time = end_time = 0
            for element in root:
                if element.tag == 'segment' and 'transcriber_start' in element.attrib and 'transcriber_end' in element.attrib:
                    start_time = float(element.attrib['transcriber_start'])
                    end_time = float(element.attrib['transcriber_end'])
                    if start_time is not None and end_time is not None:
                        timestamps.append([start_time, end_time])
        timestamps.sort()
        return timestamps


    def label_frames(self, directory, FRAME_LENGTH):
        if 'ground_truth' not in self.data:
            frame_count = len(self.data['frames'])
            dt = np.dtype(np.uint8)
            self.data.create_dataset('ground_truth', (frame_count,),dtype=dt,data=np.zeros(frame_count))


        for i, file in enumerate(self.data['files']):
            for j, name in enumerate(self.get_track_names(directory)):
                if name in file:
                    xml_files = self.get_xml_files(directory,track_name=name)
                    timestamps = self.get_timestamps(xml_files)

                    for x,y in enumerate(timestamps):
                        start_frame = int(self.data['starting_point'][i]) + int(y[0]/(FRAME_LENGTH/1000))
                        end_frame = int(self.data['starting_point'][i]) + int(y[1]/(FRAME_LENGTH/1000))
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
        self.data_mode = 2

    def get_data(self, index_from, index_to):
        frames = self.data['frames'][index_from:index_to]
        mfcc = self.data['mfcc'][index_from:index_to]
        delta = self.data['delta'][index_from:index_to]
        labels = self.data['labels'][index_from:index_to]
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

    args = parser.parse_args()

    #Calculate frame size in data points
    FRAME_SIZE = int(args.SRate * (args.FSize / 1000.0))

    #Change path
    PATH = args.Path
    os.chdir(PATH)

    AMI_speech = FileManager('AMI_speech', 'amicorpus')
    speech_data = AMI_speech.data
    data = h5py_cache.File('data_AMI.hdf5', 'a', chunk_cache_mem_size=1024 ** 3)
    #print(extract_features(speech_data['frames'], SAMPLE_WIDTH=args.SWidth, SAMPLE_RATE=args.SRate, SAMPLE_CHANNELS=1, mfcc_window_size=args.MFCC_WinS, FRAME_SIZE_MS=args.FSize)[2])

    #AMI_speech.label_frames(PATH,FRAME_LENGTH=30)
    #AMI_speech.prepare_files(normalize=True,SAMPLE_RATE=args.SRate, SAMPLE_WIDTH=args.SWidth, SAMPLE_CHANNELS=args.SChannels)
    #print(AMI_speech.data)
    #AMI_speech.collect_frames(FRAME_SIZE=FRAME_SIZE)

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

            data['labels'][pos: pos + len(labels)]
            pos += len(labels)
            print('Preparing ({0:.2f} %)'.format((pos * 100) / total), end='\r', flush=True)
        data.flush()
        print('\nDone')




    # Test generator features.
    generator = DataGenerator(data, size_limit=10000)

    generator.setup_generation(frame_count=3, step_size=5, batch_size=4)
    #generator.set_noise_level_db('-3')
    generator.use_train_data()
    X, y = generator.get_batch(50)

    print(f'Load a few frames into memory:\n{X[0]}\n\nCorresponding label: {y[0]}')

    generator.plot_data(0, 1000)
    #print(data['frames'][0:2])

    #get_xml(PATH)


if __name__ == '__main__':
    main()