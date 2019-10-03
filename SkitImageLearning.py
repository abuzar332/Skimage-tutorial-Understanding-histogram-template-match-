from skimage import io, data, filters
import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.draw as draw
import skimage.color as color
import skimage.segmentation as seg

here_path = os.path.dirname(os.path.realpath(__file__))


filename = os.path.join(here_path, 'DesktopScreen.PNG')
camera = io.imread(filename)

image = io.imread('test2.png')  #.astype( np.float32)
image2button=io.imread("imageShortTest.PNG")
image2button2 = io.imread('WebsiteHomeButton.png')

plt.imshow(image);

from skimage.feature import match_template



result = match_template(image, image2button2)

ij = np.unravel_index(np.argmax(result), result.shape)

# x,y start from top top left

y,x = ij[:-1]



image_slic = seg.slic(image,n_segments=155)

plt.imshow(color.label2rgb(image_slic, image, kind='avg'));


image_gray=image_gray = color.rgb2gray(image) 
#image =  data.coins()  # or any NumPy array!




#plt.imshow(image_gray);https://github.com/OlgaBelitskaya/deep_learning_projects
edges = filters.sobel(image_gray)
io.imshow(edges)

ss="test"

# Load a small section of the image.
image = data.coins()[0:95, 70:370]

fig, axes = plt.subplots(ncols=2, nrows=3,
                         figsize=(8, 4))
ax0, ax1, ax2, ax3, ax4, ax5  = axes.flat
ax0.imshow(image, cmap=plt.cm.gray)
ax0.set_title('Original', fontsize=24)
ax0.axis('off')


# Histogram.
values, bins = np.histogram(image,
                            bins=np.arange(256))

ax1.plot(bins[:-1], values, lw=2, c='k')
ax1.set_xlim(xmax=256)
ax1.set_yticks([0, 400])
ax1.set_aspect(.2)
ax1.set_title('Histogram', fontsize=24)








