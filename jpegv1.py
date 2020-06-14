import numpy as np
from scipy.stats import entropy
import numpy.linalg as nl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from PIL import Image

import JPEG.Functions as jpeg


# image 8 x 8
taille = 8
image1 = np.array([np.ones(taille) for x in range(taille)])
# generate matrix 8 x 8 with each cell is 1
print(image1)

# apply DCT for this image and show it
plt.imshow(jpeg.DCT(image1).astype(int), cmap='gray')
plt.show()

# get original image by using iDCT -> show it
imageRetour = jpeg.iDCT(jpeg.DCT(image1).astype(int)).astype(int)
plt.imshow(imageRetour, cmap='gray')
plt.show()

### test code - uncommentaire to test
# plt.subplot(2, 2, 1)
# plt.plot(range(5), color = 'red')
# plt.subplot(2, 2, 2)
# plt.plot(range(5), color = 'blue')
# plt.subplot(2, 2, 3)
# plt.plot(range(5), color = 'green')
# plt.subplot(2, 2, 4)
# plt.plot(range(5), color = 'black')
# plt.show()


# create 4 images with each cell is 0
image1 = np.array([np.zeros(taille) for x in range(taille)])
image2 = np.array([np.zeros(taille) for x in range(taille)])
image3 = np.array([np.zeros(taille) for x in range(taille)])
image4 = np.array([np.zeros(taille) for x in range(taille)])
image1[1][0] = 1
image2[1][7] = 1
image3[4][1] = 1
image4[7][7] = 1
images = [image1, image2, image3, image4]

# show them -> iDCT
fig = plt.figure(1)
for x in range(1,4*3+1):
    if x%3 == 0:
        plt.subplot(4,3,x)
        plt.imshow(images[int(x/3)-1], cmap='gray')
    if (x+1)%3 == 0:
        plt.subplot(4,3,x)
        zvalues = images[int(x/3)]
        ax = fig.add_subplot(4 , 3, x, projection='3d')
        xx, yy = np.meshgrid(np.arange(taille), np.arange(taille))
        ax.plot_surface(yy,xx,jpeg.iDCT(images[int(x/3)]), cmap='gray')
    if (x+2)%3 == 0:
        plt.subplot(4,3,x)
        plt.imshow(jpeg.iDCT(images[int(x/3)]), cmap='gray')
plt.show()

# load input image
img = Image.open('./ulr.png')
print(img.shape)
plt.imshow(img, cmap='gray')
plt.show()

# img = np.array(img).astype(int)
# plt.imshow(img, cmap='gray')
# plt.show()

#apply DCT to this image

tailleX = img.shape[0]
tailleY = img.shape[1]
dct_img = np.array([np.zeros(tailleX) for x in range(tailleY)])
for x in range(0,tailleX,8):
    for y in range(0,tailleY,8):
        dct_img[x:x+8,y:y+8]=jpeg.DCT(img[x:x+8,y:y+8])
plt.imshow(dct_img, cmap='gray')
plt.show()

#reverse image by using iDCT
imgInverse = np.array([np.zeros(tailleX) for x in range(tailleY)])
for x in range(0,tailleX,8):
    for y in range(0,tailleY,8):
        imgInverse[x:x+8,y:y+8]=jpeg.iDCT(dct_img[x:x+8,y:y+8])

### count loss ratio
print("ratio de perte : ",nl.norm(img-imgInverse, 2))


# load image
img = Image.open('./Barbara.jpg')
print(img.size)#512,512
tailleX = img.size[0]
tailleY = img.size[1]
img = np.array(img.getdata()).reshape(img.size[0], img.size[1])

sum_dct = np.array([np.zeros(8) for x in range(8)])
for x in range(0,tailleX,8):
    for y in range(0,tailleY,8):
        sum_dct+=abs(jpeg.DCT(img[x:x+8,y:y+8]))
plt.imshow(sum_dct, cmap='gray')
plt.show()
sum_dct = sum_dct / np.sum(sum_dct)

fig = plt.figure(1)
ax = fig.add_subplot(1 , 1, 1, projection='3d')
xx, yy = np.meshgrid(np.arange(8), np.arange(8))
ax.plot_surface(xx,yy,sum_dct, cmap='gray')
plt.show()

print(entropy(np.reshape(sum_dct, (64))))
liste, _ = np.histogram(img, bins=256,range=(0,255))
print(liste)
plt.hist(liste, bins=256)
plt.show()


# define seuillage_DCT for ulr
def seuillage_DCT(coeff, pourcent_max):
    thres=pourcent_max*np.max(coeff)
    coeff_seuil = np.where(np.abs(coeff) > thres, coeff, 0.0)
    return coeff_seuil

img = Image.open('./ulr.png')
img = np.array(img).astype(int)
tailleX = img.shape[0]
tailleY = img.shape[1]
def compression(img, coeff):
    tailleX = img.shape[0]
    tailleY = img.shape[1]
    dct_img = np.array([np.zeros(tailleX) for x in range(tailleY)])
    for x in range(0,tailleX,8):
        for y in range(0,tailleY,8):
            dct_img[x:x+8,y:y+8]=jpeg.DCT(seuillage_DCT(img[x:x+8,y:y+8], coeff))
    return dct_img

def retrouverImage(dct_img):
    tailleX = dct_img.shape[0]
    tailleY = dct_img.shape[1]
    imgInverse = np.array([np.zeros(tailleX) for x in range(tailleY)])
    for x in range(0,tailleX,8):
        for y in range(0,tailleY,8):
            imgInverse[x:x+8,y:y+8]=jpeg.iDCT(dct_img[x:x+8,y:y+8])
    return imgInverse

# compress this image by many different compress ratios 
plt.subplot(1,4,1)
plt.imshow(retrouverImage(compression(img, 0.01)), cmap='gray')
plt.subplot(1,4,2)
plt.imshow(retrouverImage(compression(img, 0.05)), cmap='gray')
plt.subplot(1,4,3)
plt.imshow(retrouverImage(compression(img, 0.30)), cmap='gray')
plt.subplot(1,4,4)
plt.imshow(retrouverImage(compression(img, 0.90)), cmap='gray')
plt.show()

### seuillange_DCT for Barbara
def seuillage_DCT(coeff, pourcent_max):
    thres=pourcent_max*np.max(coeff)
    coeff_seuil = np.where(np.abs(coeff) > thres, coeff, 0.0)
    return coeff_seuil

img = Image.open('./Barbara.jpg')
tailleX = img.size[0]
tailleY = img.size[1]
img = np.array(img.getdata()).reshape(img.size[0], img.size[1])
def compression(img, coeff):
    tailleX = img.shape[0]
    tailleY = img.shape[1]
    dct_img = np.array([np.zeros(tailleX) for x in range(tailleY)])
    for x in range(0,tailleX,8):
        for y in range(0,tailleY,8):
            dct_img[x:x+8,y:y+8]=jpeg.DCT(seuillage_DCT(img[x:x+8,y:y+8], coeff))
    return dct_img

def retrouverImage(dct_img):
    tailleX = dct_img.shape[0]
    tailleY = dct_img.shape[1]
    imgInverse = np.array([np.zeros(tailleX) for x in range(tailleY)])
    for x in range(0,tailleX,8):
        for y in range(0,tailleY,8):
            imgInverse[x:x+8,y:y+8]=jpeg.iDCT(dct_img[x:x+8,y:y+8])
    return imgInverse

plt.subplot(1,4,1)
plt.imshow(retrouverImage(compression(img, 0.01)).astype(int), cmap='gray')
plt.subplot(1,4,2)
plt.imshow(retrouverImage(compression(img, 0.05)).astype(int), cmap='gray')
plt.subplot(1,4,3)
plt.imshow(retrouverImage(compression(img, 0.30)).astype(int), cmap='gray')
plt.subplot(1,4,4)
plt.imshow(retrouverImage(compression(img, 0.90)).astype(int), cmap='gray')
plt.show()





