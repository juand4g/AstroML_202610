from matplotlib import pyplot as plt

def show_log_hist(img,bin_num=500):
    plt.hist(img.ravel(),bins=bin_num,log=True)
    plt.show()

def show_img(img, map="magma"):
    plt.imshow(img, cmap=map, norm="log")
    plt.colorbar()
    plt.show()

def show_magnitudes(img,mags,centroids,map="inferno"):
    plt.imshow(img, cmap=plt.cm.gray, norm="log")
    sc = plt.scatter(centroids[:,0],   # x
                    centroids[:,1],   # y
                    c=mags,           # color según flujo
                    s=20,
                    cmap=map)

    plt.colorbar(sc, label='Magnitud instrumental')
    plt.xlim(0, img.shape[1])
    plt.ylim(0, img.shape[0])

    plt.show()
