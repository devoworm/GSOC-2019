package org.deeplearning4j;


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.awt.image.RenderedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.ImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.api.shape.Shape;

public class Keras_import {
	private static Logger log = LoggerFactory.getLogger(Keras_import.class);
	public static void main(String[] args) throws Exception{
		
		
		int height = 28;
		int width = 28;
		INDArray final_output;
		INDArray final_output2;
		
		String fullModel = new ClassPathResource("keras_mnist_autoencoder.h5").getFile().getPath();
		ComputationGraph model = KerasModelImport.importKerasModelAndWeights(fullModel, true);
		
		
//		String modelJson = new ClassPathResource("keras_mnist_autoencoder_model.json").getFile().getPath();
//		ComputationGraphConfiguration modelConfig = KerasModelImport.importKerasModelConfiguration(modelJson);
//		
//		
//		String modelWeights = new ClassPathResource("keras_mnist_autoencoder_weights.h5").getFile().getPath();
//		ComputationGraph network = KerasModelImport.importKerasModelAndWeights(modelJson, modelWeights);
//
//		System.out.println(model.conf());
//		System.out.println(modelConfig);
//		System.out.println(network.conf());
		
		String filepath = "/home/roronoa/Desktop/GSOC2019/INCF/experiments/W-net/x_test[0].png";
		ConvertToGray(filepath);
		
		File img = new File(filepath); // loading image
		System.out.println(img.getClass().getName()+ "  Type of img");
		
		
		NativeImageLoader loader = new NativeImageLoader(height, width, 1);
		INDArray input_image = loader.asMatrix(img);
		DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
		scaler.transform(input_image);
		INDArray[] output = model.output(input_image);
		final_output = output[0];
		
		INDArray twod = final_output.reshape('c',final_output.size(2), final_output.size(3));
		
		
		System.out.println(input_image.getClass().getName()+"  Input image");
		System.out.println(final_output.getClass().getName()+"  Output image");
		
		System.out.println(input_image.length()+"  Input image length");
		System.out.println(final_output.length()+"  Output image length");
		
		System.out.println(Shape.shapeToString(final_output)+"  Shape of output image");
		System.out.println(Shape.shapeToString(twod)+"  Shape of reshaped image");
		
		System.out.println(twod);
		
		ImageLoader.toImage(twod);
	}
	
	
	public static void ConvertToGray(String filepath) throws Exception
	{
        File input = new File(filepath);
        BufferedImage image = ImageIO.read(input);
        int width = image.getWidth();
        int height = image.getHeight();
        
        for(int i=0; i<height; i++) {
        
           for(int j=0; j<width; j++) {
           
              Color c = new Color(image.getRGB(j, i));
              int red = (int)(c.getRed() * 0.299);
              int green = (int)(c.getGreen() * 0.587);
              int blue = (int)(c.getBlue() *0.114);
              Color newColor = new Color(red+green+blue,
              
              red+green+blue,red+green+blue);
              
              image.setRGB(j,i,newColor.getRGB());
           }
        }
        System.out.println(image.getClass().getName()+ "   gray image type");
	}

}