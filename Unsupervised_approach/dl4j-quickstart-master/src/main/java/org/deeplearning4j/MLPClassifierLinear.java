package org.deeplearning4j;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

//////////////////////////////////////
// This is the version of MLPClassifierLinear for the screencast
// This example can also be found as part of a large collection 
// of examples at https://github.com/deeplearning4j/dl4j-examples.git
// with instructions on configuring your environment here
// http://deeplearning4j.org/quickstart
// More information at http://skymind.io/
//////////////////////////////////////

public class MLPClassifierLinear {
    public static void main(String[] args) throws Exception{
	int seed = 123;
        double lr = 0.01;
        int batchSize = 50;
        int nEpochs = 30;
        int numInputs = 2;
        int numOutputs = 2;
        int numHiddenNodes = 20;

        //load the training data
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File("linear_data_train.csv")));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,batchSize,0,2);

        // load the test-evaluation data:

        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File("linear_data_eval.csv")));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest,batchSize,0,2);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
	    .seed(seed)
	    .maxNumLineSearchIterations(1)
	    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
	    .updater(new Nesterovs(lr, 0.9))
	    .list()
	    .layer(0,new DenseLayer.Builder()
		   .nIn(numInputs)
		   .nOut(numHiddenNodes)
		   .weightInit(WeightInit.XAVIER)
		   .activation(Activation.RELU)
		   .build())
	    .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
		   .weightInit(WeightInit.XAVIER)
		   .activation(Activation.SOFTMAX)
		   .weightInit(WeightInit.XAVIER)
		   .nIn(numHiddenNodes)
		   .nOut(numOutputs)
		   .build()
		   )
	    .pretrain(false).backprop(true).build();

        // System.out.println(conf.toJson());
	MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        for (int n = 0; n < nEpochs; n++){
            model.fit(trainIter);
        }

        System.out.println("Evaluate model.......");
        Evaluation eval = new Evaluation(numOutputs);
        while(testIter.hasNext()){
            DataSet t = testIter.next();
            INDArray features = t.getFeatures();
            INDArray lables = t.getLabels();
            INDArray predicted = model.output(features,false);
            eval.eval(lables,predicted);
        }
        System.out.println(eval.stats());
    }
}
