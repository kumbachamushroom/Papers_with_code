# The Cluster_Embeddings.py script is used to extract embeddings from a given list
# of speech samples and to cluster the embeddings according to a given algorithm
# to be used for clustering experiments
from Triplet_Net import VGGVox
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torchaudio
from Triplet_DataLoader import Single_Speaker_Loader
import argparse
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import random
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
sns.set_style("darkgrid")
from sklearn.cluster import KMeans, SpectralClustering

def get_embeddings(DataLoaders, model):
    model.eval()
    embeddingList = []
    with torch.no_grad():
        for Loader in DataLoaders:
            i = 0
            for batch in iter(Loader):
                i = i + 1
                spectrograms, _, string_labels = batch
                #print(string_labels)
                spectrograms = spectrograms.cuda()
                embeddings = model(spectrograms)
                embeddingList.append((embeddings,string_labels))
                print("Labelling : {}/{} ".format(i,len(iter(Loader))), end='\r', flush=True)
    return embeddingList

def t_SNE(embeddings,speaker_labels, num_dimensions = 2):
    """
    t_SNE reduces the high dimensional speaker embeddings into lower dimension representations for use in plotting and clustering
    :param embeddings: embedding tensor of size [# of embeddings, # of dimensions]
    :param num_dimensions: number of dimensions to reduce to (default = 2)
    :return:
    """
    unique_labels = list(set(speaker_labels))
    print(unique_labels)
    colors = ['blue', 'red', 'orange', 'green', 'yellow']

    color_list = []
    for label in speaker_labels:
        color_list.append(colors[unique_labels.index(label)])
    print(color_list)



    embeddings = embeddings.cpu().numpy()
    coords = TSNE(n_components=num_dimensions, random_state=47, perplexity=20, n_iter=4000).fit_transform(embeddings)
    kmeans = KMeans(n_clusters=len(unique_labels))
    kmeans.fit(coords)
    y_kmeans = kmeans.predict(coords)

    spectral = SpectralClustering(n_clusters=len(unique_labels), affinity='nearest_neighbors',
                                  assign_labels='kmeans')
    spectral_labels = spectral.fit_predict(coords)

    coords = pd.DataFrame({'x' : [x for x in coords[:,0]],
                           'y' : [y for y in coords[:,1]],
                           'labels' : speaker_labels,
                           'color' : color_list,
                           'k-means' : kmeans,
                           'spectral' : spectral_labels})
    print(coords)

    fig, _ = plt.subplots()
    fig.set_size_inches(22, 12)

    # Basic plot
    p1 = sns.scatterplot(x="x",
                         y="y",
                         hue="color",
                         s=1,
                         legend=None,
                         # scatter_kws={'s': 1},
                         data=coords)
    custom = [Line2D([], [], marker='.', color='blue', linestyle='None', markersize=8),
              Line2D([], [], marker='.', color='orange', linestyle='None', markersize=8)]

    legend = plt.legend(custom, ['True Detection', 'False Detection'], loc='upper right', title="Detection type",
                        fontsize=15)
    plt.setp(legend.get_title(), fontsize=20)
    # Adds annotations one by one with a loop
    for line in range(0, coords.shape[0]):
        p1.text(coords["x"][line],
                coords['y'][line],
                '  ' + coords["labels"][line].title(),
                horizontalalignment='left',
                verticalalignment='bottom', size='small',
                color=coords['color'][line],
                fontsize=1,
                weight='normal'
                ).set_size(10)

    #plt.xlim(coords[:, 0].min(), coords[:, 0].max())
    #plt.ylim(coords[:, 1].min(), coords[:, 1].max())

    plt.title('Visualizing Word Embeddings using t-SNE fastText SG - window 21', fontsize=24)
    # lgnd=plt.legend(fontsize=50)
    # lgnd.legendHandles[0]._legmarker.set_markersize(6)
    # plt.setp(p1._legend.get_texts(), fontsize=16)
    plt.tick_params(labelsize=20)
    plt.xlabel('tsne-one', fontsize=20)
    plt.ylabel('tsne-two', fontsize=20)
    plt.show()

    fig, _ = plt.subplots()
    fig.set_size_inches(22, 12)
    plt.scatter(x=coords["x"], y=coords["y"], c=y_kmeans, s=50, cmap='viridis')
    plt.setp(legend.get_title(), fontsize=20)
    # Adds annotations one by one with a loop
    for line in range(0, coords.shape[0]):
        plt.text(coords["x"][line],
                coords['y'][line],
                '  ' + coords["labels"][line].title(),
                horizontalalignment='left',
                verticalalignment='bottom', size='small',
                color='purple',
                fontsize=1,
                weight='normal'
                ).set_size(10)

    centers = kmeans.cluster_centers_
    plt.scatter(centers[:,0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()

    fig, _ = plt.subplots()
    fig.set_size_inches(22, 12)
    plt.scatter(x=coords["x"], y=coords["y"], c=spectral_labels, s=50, cmap='viridis')
    plt.show()

    return coords



def coverage_and_purity(df, method='k-means'):
    speaker_labels = df['labels'].values
    print(speaker_labels)
    clusters = df[method].values
    print(clusters)
    for i in list(set(clusters)):
        print("Now looking at cluster ", i)
        labels = df[df.method == i]
        print(df['labels'].value_counts())







#def get_average_embedding()


def main():
    global args
    parser = argparse.ArgumentParser(description='VGG Embeddings Clustering Script')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='input batch size for embeddings')
    parser.add_argument('--s-file',
                        default='/home/lucas/PycharmProjects/Papers_with_code/data/AMI/amicorpus_individual/Extracted_Speech/trimmed_sample_list.txt',
                        type=str,
                        help='path to sample list')
    parser.add_argument('--load-path',default='/home/lucas/PycharmProjects/Papers_with_code/data/models/VGG_Triplet'
                        ,type=str, help='path to load models from')
    parser.add_argument('--name', default='Cluster Embeddings', type=str,
                        help='name of experiment')
    parser.add_argument('--speakers', type=list, default=['FTD019UID','MTD017PM','MTD018ID','MTD020ME'],
                        metavar='N', help='List of speaker labels to extract embeddings from')
    #default=['FEO065', 'FEE066','MEE067','MEE068','MEO069']
    #default=['FTD019UID','MTD017PM','MTD018ID','MTD020ME']
    args = parser.parse_args()
    #No point in running this on CPU
    device = torch.device("cuda:0")

    DataLoaders = []
    kwargs = {'num_workers': 2, 'pin_memory': True}

    #Create a List of DataLoaders, seperate DataLoader for each speaker
    for speaker in args.speakers:
        DataLoaders.append(torch.utils.data.DataLoader(Single_Speaker_Loader(path=args.s_file, speaker=speaker),batch_size=args.batch_size, shuffle=False, **kwargs))

    model = VGGVox()
    model.to(device)
    model.load_state_dict(torch.load(args.load_path))
    cudnn.benchmark = True
    embeddingList = get_embeddings(DataLoaders=DataLoaders, model=model)
    tensor_length = [len(embedding[0]) for embedding in embeddingList]
    tensor = torch.empty(sum([len(embedding[0]) for embedding in embeddingList]), 256)
    print(tensor.size())
    embeddings = [embedding[0] for embedding in embeddingList]
    embeddings = torch.cat(embeddings, dim=0)
    #labels = zip([embedding[1] for embedding in embeddingList])
    labels = []
    for listing in [embedding[1] for embedding in embeddingList]:
        for item in listing:
            labels.append(item)

    #labels =[item for item in [list for list in [embedding[1] for embedding in embeddingList]]]
    coords = t_SNE(embeddings, speaker_labels=labels, num_dimensions=2)
    #scatter_plot(coords = t_SNE(embeddings,speaker_labels=labels, num_dimensions=3), dimensions=3)

    coverage_and_purity(df=coords, method='k-means')












if __name__ == "__main__":
    main()