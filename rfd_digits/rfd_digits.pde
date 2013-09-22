// first we import the OpenCV library for Processing. We need this to setup 
// the correct system library paths
import gab.opencv.*;
import java.util.Arrays;
// Import all the classes that we need from the OpenCV Java library. A reference
// for these classes can be found here: http://docs.opencv.org/java/
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.TermCriteria;
import org.opencv.ml.CvRTParams;
import org.opencv.ml.CvRTrees;

// we use this variable to know how many different answers we have. In our case, we have 10 number, 0-9.
int NUMBER_OF_CLASSES = 10;

RandomForest forest;

// Create an array of labeled samples from a Processing Table:
// Assumes each row represents 1 sample
// and the last column in each row is the class label
Sample[] samplesFromTable(Table table){
   Sample[] result = new Sample[table.getRowCount()];
  
   for(int row = 0; row < table.getRowCount(); row++){
    Sample sample = new Sample(table.getColumnCount()-1);
  
    for(int col = 0; col < table.getColumnCount()-1; col++){
      sample.featureVector[col] = table.getInt(row, col);
    }
    
    int label = table.getInt(row, table.getColumnCount()-1);
    sample.setLabel(label);
    
    result[row] = sample;
  }
  
  return result;
}

void setup(){
  // just do this to load opencv lib
  OpenCV opencv = new OpenCV(this, 0,0);

  forest = new RandomForest(this);

  
  Table trainingData = loadTable("training.csv");
  forest.addTrainingSamples(samplesFromTable(trainingData));
  forest.train();
 
  Table testData = loadTable("testing.csv");
  Sample[] testSamples = samplesFromTable(testData);
  
  int numCorrect = 0;
  for(int i = 0; i < testSamples.length; i++){
    Sample sample = testSamples[i];
    
    double prediction = forest.predict(sample);
    
    println("Sample "+i + " was predicted to be: " + prediction);
    
    if((int)prediction == sample.label){
      numCorrect++;
    }
  }
  
  println("Score: " + numCorrect + "/" + testSamples.length + " (" + ((float)numCorrect/testSamples.length) + "%)" );
}
