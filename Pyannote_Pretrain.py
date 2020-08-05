import torch
from pyannote.database import get_protocol
from pyannote.database import FileFinder
# speech activity detection model trained on AMI training set
sad = torch.hub.load('pyannote/pyannote-audio', 'sad_ami')
# speaker change detection model trained on AMI training set
scd = torch.hub.load('pyannote/pyannote-audio', 'scd_ami')
# overlapped speech detection model trained on AMI training set
ovl = torch.hub.load('pyannote/pyannote-audio', 'ovl_ami')
# speaker embedding model trained on AMI training set
emb = torch.hub.load('pyannote/pyannote-audio', 'emb_ami')

preprocessors = {'audio': FileFinder}
protocol = get_protocol('AMI.SpeakerDiarization.MixHeadset',preprocessors=preprocessors)
test_file = next(protocol.test())
