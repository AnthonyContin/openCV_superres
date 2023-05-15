import cv2
import matplotlib.pyplot as plt
from cv2 import dnn_superres


sr = cv2.dnn_superres.DnnSuperResImpl_create()
# Image
img = cv2.imread("nissan.jpg")

# read the model
path = "EDSR_x4.pb"
sr.readModel(path)

# # set the model and scale
# sr.setModel("fsrcnn", 3)
#
# # upsample the image
#
# upscaled = sr.upsample(img)
#
# # save the upscaled image
# cv2.imwrite('uncarroL.jpg', upscaled)

sr.setModel("edsr", 4)

result = sr.upsample(img)

# Resized image
resized = cv2.resize(img, dsize=None, fx=3, fy=3)

plt.figure(figsize=(12, 8))
plt.subplot(1, 3, 1)
# Original image
plt.imshow(img[:, :, ::-1])
plt.subplot(1, 3, 2)
# SR upscaled
plt.imshow(result[:, :, ::-1])
plt.subplot(1, 3, 3)
# OpenCV upscaled
plt.imshow(resized[:, :, ::-1])
plt.show()