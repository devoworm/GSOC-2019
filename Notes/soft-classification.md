Each pixel will be part of a larger set of categories like the type of cells, emerging organs, cell lineage, or cell position. The labels consist of what you might call "soft attributes". That is, each pixel has a membership in each category. Alternatively, a small group of adjacent cells (a square or a column with a known spatial location) will have an average membership. That provides descriptive information, which allows us to go beyond simply segmenting shapes as "cells", or labelling a pixel with minimal info (x,y,z,t,theta).

* soft margins in SVM (edges of cells membership function between 0,1).

* soft classification more generally (different categories used to determine membership function of a pixel 
between 0,1).
