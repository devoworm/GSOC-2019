package com.mycompany.imagej;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.gui.GenericDialog;
import ij.plugin.filter.PlugInFilter;
import ij.process.ImageProcessor;

import org.nd4j.imports.graphmapper.tf.TFGraphMapper;


import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.io.ClassPathResource;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;



public class Process_Pixels implements PlugInFilter {
	protected ImagePlus image;
	private static SameDiff sd;

	public double value;
	public String name;

	@Override
	public int setup(String arg, ImagePlus imp) {
		if (arg.equals("about")) {
			showAbout();
		}

		image = imp;
		return DOES_8G | DOES_16 | DOES_32 | DOES_RGB;
	}

	@Override
	public void run(ImageProcessor ip) {
			try {
				process(ip);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
	}
	

	// load the tensorflow model.
	public static void loadModel(String filepath) throws Exception{
        File file = new File(filepath);
        if (!file.exists()){
            file = new ClassPathResource(filepath).getFile();
        }

        sd = TFGraphMapper.getInstance().importGraph(file);

        if (sd == null) {
            throw new Exception("Error loading model : " + file);
        }
    }
	
	// resize a buffered image
	public static BufferedImage resize(BufferedImage img, int height, int width) {
        Image tmp = img.getScaledInstance(width, height, Image.SCALE_SMOOTH);
        BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g2d = resized.createGraphics();
        g2d.drawImage(tmp, 0, 0, null);
        g2d.dispose();
        return resized;
    }
	
	// pass the input image to the model and get the output
	public static INDArray predict (BufferedImage input_b) throws IOException{


        BufferedImage bimage = input_b;

        BufferedImage img = resize(bimage, 224, 224);

        
        float data[] = new float[img.getWidth() * img.getHeight()];
        
        for(int i = 0; i < img.getWidth(); i++){
            for(int j = 0; j < img.getHeight(); j++){
            	int p = (img.getRGB(i, j));

                int r = (p>>16)&0xff; 
                int g = (p>>8)&0xff; 
                int b = p&0xff;

                float greyScale = (r + g + b) / 3;
                greyScale /= 255.;
                data[i * img.getWidth() + j] = greyScale;
            }
        }
        
        
        
        INDArray arr = Nd4j.create(data).reshape(1, 224, 224, 1);

        sd.associateArrayWithVariable(arr, sd.variables().get(0));
        
        INDArray output = sd.execAndEndResult();
        
        return output;
    }
	
	// driver method
	public void process(ImageProcessor ip) throws Exception{
		BufferedImage output_b = null;
		BufferedImage input_b = ip.getBufferedImage();
		
		
		int original_x = input_b.getHeight();
		int original_y = input_b.getWidth();
		
    	
        loadModel("/home/roronoa/Desktop/GSOC2019/INCF/experiments/Wnet/wnetmodel_1.13.1.pb");
        
        
        INDArray prediction = predict(input_b);
        
        
        prediction = prediction.reshape('c', 224, 224, 3);
        

        
        float[][][] out = new float[(int) prediction.size(0)][0][0];
        for(int i=0; i<(int) prediction.size(0); i++ ){
            out[i] = prediction.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all()).toFloatMatrix();
        }
        
       
        int xLength = out.length;
        int yLength = out[0].length;
        int zLength = out[0][0].length;
        
        output_b = new BufferedImage(xLength, yLength, 4);
        
        for(int x = 0; x < xLength; x++) {
            for(int y = 0; y < yLength; y++) {
                for(int z = 0; z < zLength; z++)
                {
                	out[x][y][z] = out[x][y][z]*255;
                }
            }
        }
        
       
        
        for(int x = 0; x < xLength; x++) {
            for(int y = 0; y < yLength; y++) {
                	int red = (int) out[x][y][0];
                	int green = (int) out[x][y][1];
                	int blue = (int) out[x][y][2];
                	
                	int rgb = red;
                	rgb = (rgb << 8) + green;
                	rgb = (rgb << 8) + blue;
                	output_b.setRGB(x, y, rgb);
            }
        }
        File outputfile = new File("output.png");
        
        
        output_b = resize(output_b, original_x, original_y);
        
                
        ImagePlus output_imageplus = new ImagePlus("Segmented Output", output_b);
        
        output_imageplus.show();
        
        ImageIO.write(output_b, "png", outputfile);
        
	}


	public void showAbout() {
		IJ.showMessage("ProcessPixels",
			"The plugin will segment your SPIM Image. Click \"OK\" to begin. It may take upto 60 sec of time."
		);
	}


	
	public static void main(String[] args) throws Exception {

		Class<?> clazz = Process_Pixels.class;

		new ImageJ();

		ImagePlus image = IJ.openImage();
		image.show();

		IJ.runPlugIn(clazz.getName(), "about");
	}
}