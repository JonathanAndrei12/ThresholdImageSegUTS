import numpy as np
import cv2
from skimage.util import random_noise
from matplotlib import pyplot as plt

rawImage = cv2.imread("NoiseImage.JPG")

rawHist = cv2.calcHist([rawImage],[0],None,[255],[0,255])
plt.plot(rawHist)
plt.title("Image Histogram")
plt.show()

while True:
    #Adding noises to image
    snpAdded = random_noise(rawImage, mode= 's&p', seed=None, clip=True)
    gaussianAdded = random_noise(rawImage, mode= 'gaussian', seed=None, clip=True)

    cv2.imwrite("NoiseAdded/snpAdded.JPG", np.array(255*snpAdded, dtype= np.uint8))
    cv2.imwrite("NoiseAdded/gaussianAdded.JPG", np.array(255*gaussianAdded, dtype= np.uint8))

    snpImage = cv2.imread("NoiseAdded/gaussianAdded.JPG")
    gaussianImage = cv2.imread("NoiseAdded/snpAdded.JPG")

    #Convert image color from RGB to grayscale
    rawGray = cv2.cvtColor(rawImage, cv2.COLOR_RGB2GRAY)
    snpGray = cv2.cvtColor(snpImage, cv2.COLOR_RGB2GRAY)
    gaussianGray = cv2.cvtColor(gaussianImage, cv2.COLOR_RGB2GRAY)

    cv2.imwrite("Grayscaled/rawGray.JPG", rawGray)
    cv2.imwrite("Grayscaled/snpGray.JPG", snpGray)
    cv2.imwrite("Grayscaled/gaussianGray.JPG", gaussianGray)

    rawImage2 = cv2.imread("Grayscaled/rawGray.JPG")
    snpImage2 = cv2.imread("Grayscaled/snpGray.JPG")
    gaussianImage2 = cv2.imread("Grayscaled/gaussianGray.JPG")

    #Threshold Image Segmentating
    ret, rawThreshold = cv2.threshold(rawImage2, 100, 255, cv2.THRESH_BINARY)
    ret, snpThreshold = cv2.threshold(snpImage2, 100, 255, cv2.THRESH_BINARY)
    ret, gaussianThreshold = cv2.threshold(gaussianImage2, 100, 255, cv2.THRESH_BINARY)

    cv2.imwrite("Thresholded/rawThresholded.JPG", rawThreshold)
    cv2.imwrite("Thresholded/snpThresholded.JPG", snpThreshold)
    cv2.imwrite("Thresholded/gaussianThresholded.JPG", gaussianThreshold)

    #Show Raw comparement
    rawArray = np.concatenate((rawImage, rawImage2, rawThreshold), axis= 1)
    cv2.imshow("Raw Noise", rawArray)

    #Show S&P comparement
    snpArray = np.concatenate((snpImage, snpImage2, snpThreshold), axis= 1)
    cv2.imshow("Salt & Pepper Noise", snpArray)

    #Show Gaussian comparement
    gaussianArray = np.concatenate((gaussianImage, gaussianImage2, gaussianThreshold), axis= 1)
    cv2.imshow("Gaussian Noise", gaussianArray)

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()