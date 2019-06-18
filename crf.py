import numpy as np
import cv2
from pydensecrf import densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral

image = cv2.imread('./Unsupervised_approach/test_images/a184.jpg') #original test image
softmax = cv2.imread('./Unsupervised_approach/test_images/soft-cut-channel-1.png') # encoder output of test image with soft-cut-normalized loss 

unary = unary_from_softmax(softmax).reshape(softmax.shape[0], -1)
bilateral = create_pairwise_bilateral(sdims=(25, 25), schan=(0.05, 0.05), img=image, chdim=0)

crf = dcrf.DenseCRF2D(image.shape[2], image.shape[1], softmax.shape[0])
crf.setUnaryEnergy(unary)
crf.addPairwiseEnergy(bilateral, compat=100)
pred = crf.inference(niter)
final = np.array(pred).reshape((-1, image.shape[1], image.shape[2]))
cv2.imwrite("after_crf.png", final)

# reference https://github.com/fkodom/wnet-unsupervised-image-segmentation