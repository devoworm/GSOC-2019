
## Using soft attributes for exploiting bounding boxes
### Introduction and About the data
Pixel level labeling requires a lot of effort and time, and does not always provide great results. A lot of data that we have is partially labeled, where instead of each pixel being given a 1/0 label each pixel has a membership in a macro category like the cell lineage tree or a developing organ. The data provides spatial information about these categories in the form of (x,y,z) coordinates or r theta(polar format). One way of approaching the problem is broadly defining divisions in the form of boxes and refining the boxes using a deep CNN to get a semantically segmented image.

### Method
.This technique uses region proposal methods to generate segmentation masks. The candidate segments are used to update the deep CNN. The semantic features learned by the network are used to generate better candidates. This procedure is iterated. This method can be applied to the c.elegans dataset by utilizing the spatial locations and making the bounding boxes. With ground-truth bounding boxes, we can find the candidate masks that overlap the most with the bounding boxes. This is done with an overlapping objective function. We then develop an error/cost function to maximize the overlap.

paper:https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Dai_BoxSup_Exploiting_Bounding_ICCV_2015_paper.pdf
