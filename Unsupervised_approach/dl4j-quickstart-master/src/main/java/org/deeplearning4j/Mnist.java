package org.deeplearning4j;

import org.nd4j.imports.graphmapper.tf.TFGraphMapper;


import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.io.ClassPathResource;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class Mnist {
    private static SameDiff sd;

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

    public static BufferedImage resize(BufferedImage img, int height, int width) {
        Image tmp = img.getScaledInstance(width, height, Image.SCALE_SMOOTH);
        BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g2d = resized.createGraphics();
        g2d.drawImage(tmp, 0, 0, null);
        g2d.dispose();
        return resized;
    }
    
    
    public static INDArray predict (String filepath) throws IOException{
        File file = new File(filepath);
        if (!file.exists()){
            file = new ClassPathResource(filepath).getFile();
        }

        BufferedImage image = ImageIO.read(file);

        BufferedImage img = resize(image, 224, 224);
        System.out.println(img.getWidth());
        System.out.println(img.getHeight());
        
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
        
        
        System.out.println("converted to grayscale and rescaled to [0..1]");
        
        
        INDArray arr = Nd4j.create(data).reshape(1, 224, 224, 1);

        sd.associateArrayWithVariable(arr, sd.variables().get(0));
        
        INDArray output = sd.execAndEndResult();
        
        return output;
    }


    public static void main(String[] args) throws Exception{
    	
    	
        loadModel("/home/roronoa/Desktop/GSOC2019/INCF/experiments/Wnet/wnetmodel_1.13.1.pb");
        String file = "/home/roronoa/Desktop/GSOC2019/INCF/experiments/Wnet/a184.jpg";
        
        
        INDArray prediction = predict(file);
        
        
        System.out.println("prediction came out");
        
        prediction = prediction.reshape('c', 224, 224, 3);
        
        System.out.println(prediction);
        
        float[][][] out = new float[(int) prediction.size(0)][0][0];
        for(int i=0; i<(int) prediction.size(0); i++ ){
            out[i] = prediction.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all()).toFloatMatrix();
        }
        
        System.out.println(out);

        int xLength = out.length;
        int yLength = out[0].length;
        int zLength = out[0][0].length;
        BufferedImage b = new BufferedImage(xLength, yLength, 4);
        
        System.out.println(b.getClass().getName()+"  buff image");
        
        
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
                	b.setRGB(x, y, rgb);
            }
        }
        File outputfile = new File("/home/roronoa/Desktop/GSOC2019/INCF/experiments/Wnet/output.png");
        
        
        ImageIO.write(b, "png", outputfile);
        System.out.println("done writing");

    }
}