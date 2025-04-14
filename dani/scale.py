import cv2 as cv

img = cv.imread('imagens/0/Anger.jpg')
cv.imshow('Image', img)

def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimentions = (width, height)

    return cv.resize(frame, dimentions, interpolation=cv.INTER_AREA)

cv.waitKey(0)