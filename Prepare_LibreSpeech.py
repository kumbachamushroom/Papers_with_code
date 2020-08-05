import webrtcvad
from pydub import AudioSegment
import io
import os
import glob

class Prepare_LibriSpeech:
    '''
    Combines snippets of different speakers
    '''
    def __init__(self,directory):
        self.directory = directory
        os.chdir(directory)
        files = glob.glob(directory+'/**/*.flac', recursive=True)
        self.files = files

    def create_track(self, common_speaker,sp_target_time,sp_diff_time,min_sp_time):



test = Prepare_LibriSpeech(directory=os.path.join(os.getcwd(),"data","LibriSpeech"))