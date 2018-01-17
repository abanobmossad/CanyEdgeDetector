# Import things 
from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt
import math

sigma = 0.9
high_threshold = 0.9
low_threshold = 0.2

# read the image
I = misc.imread('test_images/lena.jpg', mode='L')
print('Applying blur filter...')
I = np.double(ndimage.gaussian_filter(I, sigma))
# save and show the image
plt.figure('1')
plt.axis('off')
plt.imshow(I, cmap='gray')
plt.title('Original image')
plt.imsave(fname='output/1_Original_image.jpg', cmap='gray', arr=I, format='jpg')
# gradient in x axis
print('Applying gradient in x direction...')
Sx = ndimage.gaussian_filter(ndimage.sobel(I, axis=0), sigma)
# gradient in y axis
print('Applying gradient in y direction...')
Sy = ndimage.gaussian_filter(ndimage.sobel(I, axis=-1), sigma)

# magnitude
print('Get magnitude of x and y with B(Gx+Gy)...')
S = np.abs(np.sqrt(Sx ** 2 + Sy ** 2))
Theta = (np.arctan2(Sx, Sy)) * 180 / np.pi
# save the angle
plt.imsave(fname='output/3_Image_edge_angle.jpg', cmap='gray', arr=Theta, format='jpg')
# save and show the image
plt.figure('2')
plt.axis('off')
plt.imshow(S, cmap='gray')
plt.title('Magnitude image')
plt.imsave(fname='output/2_Magnitude_image.jpg', cmap='gray', arr=S, format='jpg')


# non maximum algorithm
def non_maximum(image, angle):
    dx, dy = image.shape[0], image.shape[1]

    output = np.zeros((dx, dy))
    for x in range(2, dx - 1):
        for y in range(2, dy - 1):
            # right and lift pixels
            if (-22.5 <= angle[x, y] <= 22.5) or (-157.5 > angle[x, y] >= -180):
                if image[x, y] >= image[x, y + 1] and image[x, y] >= image[x, y - 1]:
                    output[x, y] = image[x, y]
                else:
                    output[x, y] = 0
            # diagonal left
            elif (22.5 <= angle[x, y] <= 67.5) or (-112.5 > angle[x, y] >= -157.5):
                if image[x, y] >= image[x + 1, y + 1] and image[x, y] >= image[x - 1, y - 1]:
                    output[x, y] = image[x, y]
                else:
                    output[x, y] = 0
            # up an down
            elif (67.5 <= angle[x, y] <= 112.5) or (-67.5 > angle[x, y] >= -112.5):
                if image[x, y] >= image[x + 1, y] and image[x, y] >= image[x - 1, y]:
                    output[x, y] = image[x, y]
                else:
                    output[x, y] = 0
            # diagonal right
            elif (112.5 <= angle[x, y] <= 157.5) or (-22.5 > angle[x, y] >= -67.5):
                if image[x, y] >= image[x + 1, y - 1] and image[x, y] >= image[x - 1, y + 1]:
                    output[x, y] = image[x, y]
                else:
                    output[x, y] = 0
    return output


# hysteresis_threshold algorithm
def hysteresis_threshold(image, th1=0.9, th2=0.2):
    dx, dy = image.shape[0], image.shape[1]
    thresh_high = np.max(non_max_image) * th1
    thresh_low = thresh_high * th2
    for x in range(2, dx - 1):
        for y in range(2, dy - 1):
            if image[x, y] > thresh_high:
                image[x, y] = 1
            elif image[x, y] < thresh_low:
                image[x, y] = 0
            else:
                image[x, y] = 0.5
                # edge tracing
    return image


print('Applying Non_maximum suppression (edge thinning)...')
non_max_image = non_maximum(S, Theta)
# show and save the non max image
plt.figure('3')
plt.axis('off')
plt.title('After non maximum algorithm')
plt.imsave(fname='output/4_Non_maximum_image.jpg', cmap='gray', arr=non_max_image, format='jpg')
plt.imshow(non_max_image, cmap='gray')

# if you put low numbers more edges detected
print('Applying hysteresis_threshold with high_threshold=', high_threshold, ' and low_threshold=', low_threshold,
      '...')
h = hysteresis_threshold(non_max_image, high_threshold, low_threshold)
# plot the edged image
plt.figure('4')
plt.axis('off')
plt.title('After hysteresis threshold')
plt.imsave(fname='output/5_Hysteresis_threshold.jpg', cmap='gray', arr=h, format='jpg')
plt.imshow(h, cmap='gray')
print('Output final image...')
plt.show()
