## Unsupervised Semantic Image Segmentation

ImageJ plugin for unsupervised semantic image segmentation for SPIM Images of the embryogenesis process of *C. Elegans.*

## Motivation

- The initial step for getting statistical insights of the whole embryogenesis process (number of cells at time t, area of each cell, tracking the position of each cell, etc) of *C. Elegan*.

- On average, the process takes about 2-3 days to complete. Usually, researchers have to manually tag and label them. This software can automate this process.

## Built With
- [Tensorflow](https://www.tensorflow.org/)
- [DeepLearning4J](https://deeplearning4j.org/)
- [Maven](https://maven.apache.org/)
- [Java 8 open jdk](https://openjdk.java.net/install/)
- [ImageJ](https://imagej.net/Fiji/Downloads)
- [Eclipse](https://www.eclipse.org/downloads/)

## For Users
- **Requirements**
  - [ImageJ](https://imagej.net/Fiji/Downloads)
  - JAR file for this plugin
- **How to use it**
  1. Install ImageJ on your machine
  2. Download the JAR file from here.
  3. Navigate to your Fiji folder (`Fiji.app`)
  4. Place the downloaded jar file in the `Plugins` folder.
  5. Move up the directory and start ImageJ by clicking on the ImageJ icon.
  6. Under the `Process` tab, find `process pixels` option and click on it.
  7. Chose a SPIM image for which you need segmentation.
  8. The segmented output image will be displayed within a minute.

## For Developers
- **Requriements**
  - ImageJ
  - Eclispe/IntelliJ (Recommended - *easy to build maven projects*)
  - Java
  - GPU (Recommended - *for training the deep learning model*)
  - SPIM images

- **Description of files**
  - Fork this repository and navigate to `Unsupervised_approach` folder.

  - **[example-legacy-plugin](https://github.com/devoworm/GSOC-2019/tree/master/Unsupervised_approach/example-legacy-plugin)**: This folder contains the actual code which is packaged as a JAR file. This is the JAR file which we place in the plugins folder of ImageJ.
    - The `src` folder contains the source code.
      1. The `plugins.config` file under `/src/main/resources/` folder tells ImageJ where the plugin can be located.
      2. The `Process_Pixels.java` file under `/src/main/java/com/mycompany/imagej/` has the code for allowing the user to chose a test SPIM image for which the segmentation is displayed.
    - The `target` folder contains the class files. When we build the project using maven, the JAR file is stored here. This our required JAR file.
    - The `pom.xml` file handles all the dependencies (deep learning4j core files) you require.
  - **[output_images](https://github.com/devoworm/GSOC-2019/tree/master/Unsupervised_approach/output_images)**: The results from the [autoencoder](https://github.com/devoworm/GSOC-2019/blob/master/Unsupervised_approach/Wnet_implementation.ipynb) with different combinations of parameters, optimizers and loss functions.

  - **[test_images](https://github.com/devoworm/GSOC-2019/tree/master/Unsupervised_approach/test_images)**: Some image on which were passed to model for testing the segmentation.

  - **[crf.py](https://github.com/devoworm/GSOC-2019/blob/master/Unsupervised_approach/crf.py)**: This is supposed to be a post-processing step after training the autoencoder with soft-cut-loss function.

  - **[data_augmentation.py](https://github.com/devoworm/GSOC-2019/blob/master/Unsupervised_approach/data_augmentation.py)**: This file does all the data augmentation on this [data](https://github.com/devoworm/GSoC-2017/tree/master/src/data/interim/lattice-light-sheet) to increase its size and variance.

  - **[norm_cut_scikit.py](https://github.com/devoworm/GSOC-2019/blob/master/Unsupervised_approach/norm_cut_scikit.py)**: This file implements the normalized cut segmentation with the help of sklearn.

  - **[kmeans.py](https://github.com/devoworm/GSOC-2019/blob/master/Unsupervised_approach/kmeans.py)**: Apply K-means clustering on the output images to calculate the centroid points of each cluster.

- **[Wnet_implementation.ipynb](https://github.com/devoworm/GSOC-2019/blob/master/Unsupervised_approach/Wnet_implementation.ipynb)**: The python code using keras for implementing the W-net research paper and saving the model.

- **[Wnet_implementation_TF.ipynb](https://github.com/devoworm/GSOC-2019/blob/master/Unsupervised_approach/Wnet_implementation_TF.ipynb)**: The python code using Tensorflow for implementing the W-net research paper and saving the model. I had to reimplement the whole model in tensorflow due to a limitation caused by deeplearning4j when using custom layers for keras.

## Upgrade Guide:
- **For Users:** Whenever there is a new version of the JAR file, download it and just replace it with the old one in the `plugins` folder.

- **For Developers:**
  1. Make changes to the jupyter notebook by playing with hyperparameters, training with better data, etc and save the model.
  2. Pass that model into the `load_model` function in the `Process_pixels.java` file.
  3. Add/Remove pre-processing or post-processing steps to the `Process_pixels.java` file and build the project.
  4. The new JAR file will be stored in the `target` folder. Upload it to a website where the users can download it.

## Contribute
Please go through the references given below and be sure to read this README file completely. If you find a bug/have an idea/have a doubt, open an issue on the [issues](https://github.com/devoworm/GSOC-2019/issues) tab and please wait till some response.

## References

- [Embryogenesis](https://www.youtube.com/watch?v=M2ApXHhYbaw) - This is Bright Field imagery.
- [SPIM Images](https://github.com/devoworm/GSoC-2017/tree/master/src/data/interim) - Datasets
- [Devoworm](https://devoworm.weebly.com/), [Devoworm zoo](https://devoworm.github.io/devozoo.htm) and [openworm](http://openworm.org/) - Getting started in learning more abour *C. Elegans*
- [How to write an ImageJ plugin](https://www.youtube.com/watch?v=YIWpoBnnLio) - Writing your own ImageJ plugin
- [Wnet- Fully Unsupervised Semantic Image Segmentation](https://arxiv.org/pdf/1711.08506.pdf)