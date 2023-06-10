import facenet_pytorch, torch, os, math, numpy, torchvision.transforms, cv2, pandas
torch.cuda.empty_cache()
indir = 'output'
image_size = 100
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
resnet = facenet_pytorch.InceptionResnetV1(pretrained='vggface2').eval().to(device)
aligned = []
names = []
face_files = [ '%s/%s' % (indir, f) for f in os.listdir(indir) ]
for ifn in face_files:
    img = cv2.imread(ifn)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    face = torchvision.transforms.functional.to_tensor(numpy.float32(img))
    processed_tensor = (face - 127.5) / 128.0
    aligned.append(processed_tensor)
    names.append(os.path.basename(ifn))

aligned = torch.stack(aligned).to(device)
embeddings = resnet(aligned).detach().cpu()

#dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
#print(pandas.DataFrame(dists, columns=names, index=names))


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def scatter_thumbnails(data, images, zoom=0.12, colors=None):
    assert len(data) == len(images)

    # reduce embedding dimentions to 2
    x = PCA(n_components=2).fit_transform(data) if len(data[0]) > 2 else data

    # create a scatter plot.
    f = plt.figure(figsize=(22, 15))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], s=4)
    _ = ax.axis('off')
    _ = ax.axis('tight')

    plt.rcParams.update({'axes.facecolor':'black'})
    # add thumbnails :)
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    for i in range(len(images)):
        image = plt.imread(images[i])
        im = OffsetImage(image, zoom=zoom)
        bboxprops = dict(edgecolor=colors[i]) if colors is not None else None
        ab = AnnotationBbox(im, x[i], xycoords='data',
                            frameon=(bboxprops is not None),
                            pad=0.02,
                            bboxprops=bboxprops)
        ax.add_artist(ab)
    return ax


from sklearn.manifold import TSNE
# PCA first to speed it up
x = PCA(n_components=50).fit_transform(embeddings.tolist())
x = TSNE(perplexity=50, n_components=3).fit_transform(x)

_ = scatter_thumbnails(x, face_files, zoom=0.4)
plt.title('3D t-Distributed Stochastic Neighbor Embedding')
#plt.show()
plt.savefig('eingang_faceplot.png', dpi=300)
#df = pandas.DataFrame({'face': face_files[:len(embeddings)], 'embedding': embeddings})