
import numpy as np
import cv2
import matplotlib.pyplot as plt

image1 = cv2.imread("/home/roronoa/Desktop/GSOC2019/INCF/experiments/Wnet/scrn_m3_l2.png")

Z1 = image1.reshape((-1,3))

Z1 = np.float32(Z1)

# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 5, 1.0)

K1 = 2

ret, mask, center =cv2.kmeans(Z1,K1,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
print(center, "center")
print(mask.flatten().shape, "mask")

print(center[:,0])
print(center[:,1])
print(center[:,2])
	# print(len(c[0]))
res_image1 = center[mask.flatten()]
clustered_image1 = res_image1.reshape((image1.shape))


# plt.scatter(center[:,2], center[:,0], center[:,1], color='black')


cv2.imshow('Clustered_image1', clustered_image1)


cv2.waitKey(0)

# destroys all the windows that are created.
cv2.destroyAllWindows()