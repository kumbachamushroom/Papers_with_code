
#!/usr/bin/env python3

# This program takes a series of XML-formatted files and creates an
    # RTTM v1.3 (https://catalog.ldc.upenn.edu/docs/LDC2004T12/RTTM-format-v13.pdf)
# output file.

# Typical invocation:
#
#   nite_xml_to_rttm.py ~/Downloads/ami_public_manual_1.6.2/words/ES2008a.*.words.xml | sort -n -k 4 > /tmp/ES2008a.rttm
#
import math
import sys
import xml.etree.ElementTree as ET
import os
import glob
from operator import itemgetter


def get_track_names(directory):
    tracks = glob.glob(directory + '/amicorpus/*', recursive=True)
    tracks = [f[f.find('amicorpus') + 10:] for f in tracks]
    return tracks


def write_rttm_file(path_rttm,file,path_xml):
    lines = []
    xml_trees = []
    xml_files = glob.glob(path_xml+file+'.*.words.xml', recursive=True)
    speaker_names = []
    #for file in xml_files:
        #filename = str(file)
        #speaker_names.append(strfile[1:10])
        #print(file[-13:-12])
        #speaker_names.append(filename[filename.index('.')+1:filename.index('.')+2])

    #print(speaker_names)
    def convert_xml_to_rttm(filename):
        xml_file = xml_files[xml_file_index]
        file = str(xml_file)
        #print(filename)
        #speaker_name = file[file.index('.')+1:file.index('.')+2]
        speaker_name = str(xml_file)[str(xml_file).index('.')+1:str(xml_file).index('.')+2]
        #print(speaker_name)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        xml_trees.append(root)
        start_time = end_time = None
        for element in root:
            if element.tag == 'w' and 'starttime' in element.attrib and 'endtime' in element.attrib:
                if start_time is None:
                    start_time = float(element.attrib['starttime'])
                if end_time is None:
                    end_time = float(element.attrib['starttime'])  # yes, 'starttime'
                if math.isclose(end_time, float(element.attrib['starttime']), abs_tol=0.01):
                    # collapse the two
                    end_time = float(element.attrib['endtime'])
                else:
                    lines.append({'type': 'SPEAKER', 'file': filename, 'Channel': 1, 'starttime': float(start_time),
                                  'duration': float(end_time) - float(start_time), 'ortho': '<NA>', 'stype': '<NA>',
                                  'speaker': 'Speaker_' + speaker_name, 'conf': '<NA>'})
                    start_time = float(element.attrib['starttime'])
                    end_time = float(element.attrib['endtime'])
        if not ((start_time is None) or (end_time is None)):
            lines.append({'type': 'SPEAKER', 'file': filename, 'Channel': 1, 'starttime': float(start_time),
                          'duration': float(end_time) - float(start_time), 'ortho': '<NA>', 'stype': '<NA>',
                          'speaker': 'Speaker_' + speaker_name, 'conf': '<NA>'})
    for xml_file_index in (range(len(xml_files))):
        convert_xml_to_rttm(filename=file)
    os.chdir(path_rttm)
    f = open(file+'.rttm',"w+")
    for line in sorted(lines, key=lambda i:i['starttime']):
        f.write("{}\t {}\t {}\t {:0.2f}\t {:0.4f}\t {}\t {}\t {}\t {} \n".format(line['type'],line['file'],line['Channel'],line['starttime'],line['duration'],line['ortho'],line['stype'],line['speaker'],line['conf']))












tracks = get_track_names(directory='/home/lucas/PycharmProjects/Papers_with_code/data/AMI')
for track in tracks:
    write_rttm_file(path_xml='/home/lucas/PycharmProjects/Papers_with_code/data/AMI/words/',path_rttm='/home/lucas/PycharmProjects/Papers_with_code/data/AMI/rttm',file=track)
