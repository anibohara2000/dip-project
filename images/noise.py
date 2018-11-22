import numpy as np
import os
import cv2
def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 500#0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.1#0.04
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))#20*/
        #print(np.ceil(np.log2(vals)))
        vals = 2**(np.ceil(np.log2(vals)))
        noisy = image + np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy

def noisybw(noise_typ,image):
    if noise_typ == "s&p":
        row,col = image.shape
        s_vs_p = 0.5
        amount = 0.1#0.04
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))  for i in image.shape]
        out[coords] = 255

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out

def main():
    image_name = 'king'
    img = cv2.imread(image_name + '_orig.png', cv2.IMREAD_GRAYSCALE)
    #print(img)
    #noisy_img = noisy("gauss", img)
    #cv2.imwrite(image_name + '_gauss.png',noisy_img)
    noisy_img = noisybw("s&p", img)
    cv2.imwrite(image_name + '_sp.png',noisy_img)
    #noisy_img = noisy("poisson", img)
    #cv2.imwrite(image_name + '_poisson.png',noisy_img)
    #noisy_img = noisy("speckle", img)
    #cv2.imwrite(image_name + '_speckle.png',noisy_img)
main()