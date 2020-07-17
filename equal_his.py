import numpy as np
import cv2

def compute_hist(img):
    hist = np.zeros((256,), np.uint8)
    h, w = img.shape[:2]
    for i in range(h):
        for j in range(w):
            hist[img[i][j]] += 1
    return hist

def cal_equal_hist(hist):
    cumulator = np.zeros_like(hist, np.float64)
    for i in range(len(cumulator)):
        cumulator[i] = hist[:i].sum()
    new_hist = (cumulator - cumulator.min())/(cumulator.max() - cumulator.min()) * 255
    new_hist = np.uint8(new_hist)
    return new_hist

def equal_hist(img):
    hist = compute_hist(img).ravel()
    new_hist = cal_equal_hist(hist)

    h, w = img.shape[:2]
    for i in range(h):
        for j in range(w):
            img[i,j] = new_hist[img[i,j]]

    return img