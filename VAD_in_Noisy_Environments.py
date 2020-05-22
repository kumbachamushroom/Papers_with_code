import os
import torch
import glob
import re
from pydub import AudioSegment
import soundfile as sf
import h5py
import numpy as np
import webrtcvad

path = '../LibreSpeech_Corpus'
os.chdir(path)
print('Directory is set to ', os.getcwd())


# Specify the desired WAV-format.
SAMPLE_RATE = 16000
SAMPLE_CHANNELS = 1
SAMPLE_WIDTH = 2

# Name of folder to save the data files in.
DATA_FOLDER = 'data'

# Min/max length for slicing the voice files.
SLICE_MIN_MS = 1000
SLICE_MAX_MS = 5000

# Frame size to use for the labelling.
FRAME_SIZE_MS = 30

# Convert slice ms to frame size.
SLICE_MIN = int(SLICE_MIN_MS / FRAME_SIZE_MS)
SLICE_MAX = int(SLICE_MAX_MS / FRAME_SIZE_MS)

# Calculate frame size in data points.
FRAME_SIZE = int(SAMPLE_RATE * (FRAME_SIZE_MS / 1000.0))

OBJ_SHOW_PLAYABLE_TRACKS = True
OBJ_CUDA = torch.cuda.is_available()

OBJ_PREPARE_AUDIO = True
OBJ_TRAIN_MODELS = True

if OBJ_CUDA:
    print('CUDA has been enabled.')
else:
    print('CUDA has been disabled.')




