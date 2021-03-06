import gab.opencv.*;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.TermCriteria;
import org.opencv.ml.CvRTParams;
import org.opencv.ml.CvRTrees;

// this is the basic_example wrapped in a class. See basic_example for commented source code.

class Sample {
  double[] featureVector;
  int label;

   Sample(double[] featureVector, int label) {
    this.featureVector = featureVector;
    this.label = label;
  }
  Sample(int featureVectorSize){
    featureVector = new double[featureVectorSize];
  }
  
  void setLabel(int label){
    this.label = label;
  }
  
}

class RandomForest {  
  CvRTrees forest;
  ArrayList<Sample> trainingSamples;

   RandomForest(PApplet parent ) {
    trainingSamples = new ArrayList<Sample>();
  }

  void addTrainingSample(double[] featureVector, int label) {
    addTrainingSample(new Sample(featureVector, label));
  }

  void addTrainingSample(Sample sample) {
    trainingSamples.add(sample);
  }

  void addTrainingSamples(Sample[] samples){
    trainingSamples =  new ArrayList<Sample>(Arrays.asList(samples));
  }  

  void train() {  
    Mat trainingMat = new Mat(trainingSamples.size(), trainingSamples.get(0).featureVector.length, CvType.CV_32FC1);
    Mat labelMat = new Mat( trainingSamples.size(), 1, CvType.CV_32FC1);

    // load samples into training and label mats. 
    for (int i = 0; i < trainingSamples.size(); i++) {
      Sample trainingSample = trainingSamples.get(i);

      //trainingMat.put(0, i, trainingSample.featureVector);
            for(int j = 0; j < trainingSample.featureVector.length; j++){              
              trainingMat.put(i, j, trainingSample.featureVector[j]);
            }
            
      labelMat.put(i, 0, trainingSample.label);
    }

    Mat varType = new Mat(trainingMat.width()+1, 1, CvType.CV_8U );
    varType.setTo(new Scalar(0)); // 0 = CV_VAR_NUMERICAL.
    varType.put(trainingMat.width(), 0, 1); // 1 = CV_VAR_CATEGORICAL;

    // Begin magic numbers...
    // TODO: make this setable.

    CvRTParams params = new CvRTParams();
    params.set_max_depth(25);
    params.set_min_sample_count(5);
    params.set_regression_accuracy(0);
    params.set_use_surrogates(false);
    params.set_max_categories(15);
    // priors?????
    params.set_calc_var_importance(false);
    params.set_nactive_vars(4);
    params.set_term_crit(new TermCriteria(TermCriteria.MAX_ITER + TermCriteria.EPS, 100, 0.0f));

    forest = new CvRTrees();
    forest.train(trainingMat, 1, labelMat, new Mat(), new Mat(), varType, new Mat(), params);

  }

  // Use this function to get a prediction, after having trained the algorithm.

  double predict(Sample sample) {
    // create a mat for the prediction
    Mat predictionTraits = new Mat(1, sample.featureVector.length, CvType.CV_32FC1);

  //  predictionTraits.put(0, 0, sample.featureVector);

     //get traits from dna
        for(int i = 0; i < sample.featureVector.length; i++){
          predictionTraits.put(0, i, sample.featureVector[i]);
        }

    return forest.predict(predictionTraits);
  }
}

