import numpy as np
import matplotlib.pyplot as plt

def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 18))

    # We create a scatter plot.
    f = plt.figure(figsize=(32, 32))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=120,
                    c=palette[colors.astype(np.int)])
    #plt.xlim(-25, 25)
    #plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    ax[0, 0].set_title("Sine function")

    # We add the labels for each cluster.
    txts = []
    for i in range(18):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=50)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts



def unmount_batch(feature_t,img_t,labels_t):
  feature_img_label = []
  feature = []
  img = []
  labels = []
  for i in range(len(feature_t)):
    for j in range(len(feature_t[i])):
      feature.append(feature_t[i][j])
      img.append(img_t[i][j])
      labels.append(labels_t[i][j])
  return np.array(feature),np.array(img),np.array(labels)



def CreateDir(path):
        try:
                os.mkdir(path)
        except OSError as error:
                print(error)



def chebyshev(features_u,features_l):
  dist = []
  #print(dist)
  for line in range(features_u.shape[0]):
    dist.append((torch.max(torch.abs(features_u[line]-features_l),dim=1).values).numpy())
  return np.array(dist)

