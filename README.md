# GSoC-2019
## Main repo for Google Summer of Code 2019

### Schedule for proposals and acceptance:

#### May 6 (18:00 UTC):    Accepted student proposals announced  

* Congrats to Vinay Varma, who was selected as this year's student. Also welcome to Asmit Singh and Ujjwal Singh, who are also working on an allied project ([Digital Bacillaria](https://github.com/devoworm/Digital-Bacillaria)).

#### May 6 - May 27: Community Period Activities

* May 15 (4pm UTC): participate in the session on the #office-hours channel in [OpenWorm Slack](https://launchpass.com/openworm).

* participate in Open Data Day activities [link](https://github.com/devoworm/Open-Data-Day-2019). Interact with [DevoZoo](https://devoworm.github.io/) materials.

* preview the OpenWorm/DevoWorm curriculum [link](https://github.com/devoworm/OW-DW-Education).

* May 31 (4pm UTC): Project presentation (15-20 minute presentation on your project). Link to [this year's videos](https://www.youtube.com/channel/UChGTq41_rJwmZ1I4j7SezWQ).

#### May 27: Coding Period Begins!  

* check back for paper presentation -- coming soon! 

### Phase 1 Coding:
Phase 1 of the coding period for GSoC'19 began on May 27th and ended on the evaluation for Phase 1 coding period spanned from June 24th, 2019 to June 28th, 2019. 

**Initial Planning:**


* **The goal of Phase 1** is to create a [ImageJ](https://imagej.net/Welcome) plugin for segmenting the cells of **_C. Elegans_** during its embryogenesis process in an Unsupervised way. The plan was to employ a deep learning model for this task. As this plugin needs to be written in Java, it would not be optimal/easy to build the whole deep learning model in Java (because it has less amount of deep learning resources).

* We could train the deep learning model in Python using Tensorflow/Keras and save the model. We could then import that model into our plugin (basically any Java application) with the help of [DeepLearnign4J](https://deeplearning4j.org)'s flexibility for importing [Keras models](https://github.com/deeplearning4j/dl4j-examples/tree/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/modelimport/keras/basic) and [Tensorflow models](https://github.com/deeplearning4j/dl4j-examples/tree/master/tf-import-examples/src/main/java/org/nd4j/examples).

* During the Community bonding period, some work has been put into trying out [Jython](https://www.jython.org/) programming, but it has been marked as [not efficient](https://github.com/pytorch/pytorch/issues/6570#issuecomment-494526799) for this use case.

* This [Wnet paper](https://arxiv.org/pdf/1711.08506.pdf) served as an inspiration for doing semantic image segmentation in an Unsupervised manner.

* [This](https://github.com/devoworm/GSoC-2017/tree/master/src/data/interim) is the dataset that with which the deep learning model has to be trained.

Most of the things went as planned except for a couple of things which will be discussed here.

**What has been done:**

* Built an autoencoder model for segmenting the cells. Trained the model several times with a different set of parameters, loss functions, optimizers and deep learning frameworks(Tf.keras and keras) on Kaggle.

* Was able to get some nice results from the model. You can check them out [here](https://github.com/devoworm/GSOC-2019/tree/master/Unsupervised_approach/output_images). All of these images are outputs from different versions of this [model](https://github.com/devoworm/GSOC-2019/blob/master/Unsupervised_approach/Wnet_implementation.ipynb) tested on this [test image](https://github.com/devoworm/GSOC-2019/blob/master/Unsupervised_approach/test_images/a184.jpg).

* Constructed the [java pipeline](https://github.com/devoworm/GSOC-2019/tree/master/Unsupervised_approach/dl4j-quickstart-master) for importing saved models into a java application. 

* Setup a [basic ImageJ plugin](https://github.com/devoworm/GSOC-2019/tree/master/Unsupervised_approach/example-legacy-plugin) to which the java pipeline (for importing deep learning models) has to be added.

**Problems faced:**

* The soft-cut-normalized loss described in the [Wnet paper](https://arxiv.org/pdf/1711.08506.pdf) paper was hard to implement. Even after implementing it, it did not seem to be very useful for the kind of data that is being fed to the model. (Todo: paste link for the notebook containing soft-cut-loss function)

* Most of the documents and tutorial examples for deeplearnign4j are outdated.

* Had to but some additional RAM for my laptop.

* Limitation from the DeepLearnign4J side. A custom convolutional layer has been added as the last layer of the model. Currently, it is not possible to import deep learning models which contain convolutional custom layers into Java applications (ImageJ for this project) with the help of DeepLearnign4J. They only support simple lambda computations like [this](https://github.com/eclipse/deeplearning4j/blob/451dd76b50355358dc176f2b704e98c43423c5b8/deeplearning4j/deeplearning4j-modelimport/src/test/java/org/deeplearning4j/nn/modelimport/keras/e2e/KerasLambdaTest.java).

**Solutions:**

* Since soft-cut-normalized loss did not give any better results, the last layer's activation has been changed from `softmax` to `relu`. The output images that you see are a result of this action.

* There is a community for where the maintainers of DeepLearning4J help with issues regarding DeepLearning4J. They have helped navigate to the latest source code and examples for DeepLearning4J

* Bought 4GB additional RAM. Now my laptop has 8GB.

* **This is the main problem now** and currently working on this. There are Three options to go ahead.
    - **One option** is to try to convert the existing saved [Keras model](https://github.com/devoworm/GSOC-2019/blob/master/Unsupervised_approach/dl4j-quickstart-master/src/main/java/encoder_model_5_lambda_withth.h5) to a tensorflow model and see if it works. 

    - **Second option** is to go back to the implementation of the autoencoder and remove the custom layer (which is not compatible with dl4j, but also is the heart of the model) and reproduce the same results by adding an inbuilt keras convolutional layer. 

    - **Third option** is to implement the whole model in pure tensorflow and see how good the results will be.

    I will work on these three options this week and will update the progress.