import torch
import torch.nn as nn
import torch.nn.functional as F
from utlis import get_conv2d_output_shape

class VGGVox(nn.Module):
     """VGGVox implementation

     Reference
     ---------
     Arsha Nagrani, Joon Son Chung, Andrew Zisserman. "VoxCeleb: a large-scale
     speaker identification dataset."

     """

     def __init__(self, dimension=256):

         super().__init__()


         self.dimension = dimension

         h = 201  # 512 in VoxCeleb paper. 201 in practice.
         w = 481 # typically 3s with 10ms steps

         self.conv1_ = nn.Conv2d(1, 96, (7, 7), stride=(2, 2), padding=1)
         # 254 x 148 when n_features = 512
         # 99 x 148 when n_features = 201
         h, w = get_conv2d_output_shape((h, w), (7, 7), stride=(2, 2), padding=1)

         self.bn1_ = nn.BatchNorm2d(96)
         self.mpool1_ = nn.MaxPool2d((3, 3), stride=(2, 2))
         # 126 x 73 when n_features = 512
         # 49 x 73 when n_features = 201
         h, w = get_conv2d_output_shape((h, w), (3, 3), stride=(2, 2))

         self.conv2_ = nn.Conv2d(96, 256, (5, 5), stride=(2, 2), padding=1)
         # 62 x 36 when n_features = 512
         # 24 x 36 when n_features = 201
         h, w = get_conv2d_output_shape((h, w), (5, 5), stride=(2, 2), padding=1)

         self.bn2_ = nn.BatchNorm2d(256)
         self.mpool2_ = nn.MaxPool2d((3, 3), stride=(2, 2))
         # 30 x 17 when n_features = 512
         # 11 x 17 when n_features = 201
         h, w = get_conv2d_output_shape((h, w), (3, 3), stride=(2, 2))

         self.conv3_ = nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=1)
         # 30 x 17 when n_features = 512
         # 11 x 17 when n_features = 201
         h, w = get_conv2d_output_shape((h, w), (3, 3), stride=(1, 1), padding=1)

         self.bn3_ = nn.BatchNorm2d(256)

         self.conv4_ = nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=1)
         # 30 x 17 when n_features = 512
         # 11 x 17 when n_features = 201
         h, w = get_conv2d_output_shape((h, w), (3, 3), stride=(1, 1), padding=1)

         self.bn4_ = nn.BatchNorm2d(256)

         self.conv5_ = nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=1)
         # 30 x 17 when n_features = 512
         # 11 x 17 when n_features = 201
         h, w = get_conv2d_output_shape((h, w), (3, 3), stride=(1, 1), padding=1)

         self.bn5_ = nn.BatchNorm2d(256)

         self.mpool5_ = nn.MaxPool2d((5, 3), stride=(3, 2))
         # 9 x 8 when n_features = 512
         # 3 x 8 when n_features = 201
         h, w = get_conv2d_output_shape((h, w), (5, 3), stride=(3, 2))

         self.fc6_ = nn.Conv2d(256, 4096, (h, 1), stride=(1, 1))
         # 1 x 8
         h, w = get_conv2d_output_shape((h, w), (h, 1), stride=(1, 1))

         self.fc7_ = nn.Linear(4096, 1024)
         self.fc8_ = nn.Linear(1024, self.dimension)


     def forward(self, sequences):
         """Embed sequences

         Parameters
         ----------
         sequences : torch.Tensor (batch_size, n_samples, n_features)
             Batch of sequences.

         Returns
         -------
         embeddings : torch.Tensor (batch_size, dimension)
             Batch of embeddings.
         """


         x = sequences
        # x = torch.transpose(sequences, 1, 2).view(
             #30, 1, 3, 3)

         # conv1. shape => 254 x 148 => 126 x 73
         x = self.mpool1_(F.relu(self.bn1_(self.conv1_(x))))

         # conv2. shape =>
         x = self.mpool2_(F.relu(self.bn2_(self.conv2_(x))))

         # conv3. shape = 62 x 36
         x = F.relu(self.bn3_(self.conv3_(x)))

         # conv4. shape = 30 x 17
         x = F.relu(self.bn4_(self.conv4_(x)))

         # conv5. shape = 30 x 17
         x = self.mpool5_(F.relu(self.bn5_(self.conv5_(x))))

         # fc6. shape =
         x = F.dropout(F.relu(self.fc6_(x)))

         # (average) temporal pooling. shape =
         x = torch.mean(x, dim=-1)

         # fc7. shape =
         x = x.view(x.size(0), -1)
         x = F.dropout(F.relu(self.fc7_(x)))

         # fc8. shape =
         x = self.fc8_(x)

         return x

class TripletNet(nn.Module):
    def __init__(self, embeddingnet):
        super(TripletNet,self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, x, y, z):
        embedded_x = self.embeddingnet(x)
        embedded_y = self.embeddingnet(y)
        embedded_z = self.embeddingnet(z)
        dist_a = F.pairwise_distance(embedded_x,embedded_y,2)
        dist_b = F.pairwise_distance(embedded_x,embedded_z,2)
        return dist_a, dist_b, embedded_x, embedded_y, embedded_z

