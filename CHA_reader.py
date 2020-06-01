import pylangacq as pla
import os

os.chdir('/home/lucas/PycharmProjects/Papers_with_code/data/CALL_HOME/eng')
print(os.getcwd())

CHAT = pla.read_chat('0638.cha')
print(CHAT.part_of_speech_tags())

