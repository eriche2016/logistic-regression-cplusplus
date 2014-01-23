/***********************************************************************************
* Logistic Regression classifier version 0.03
* Implemented by Jinghui Xiao (xiaojinghui@gmail.com or xiaojinghui1978@qq.com)
* Last updated on 2014-1-17
***********************************************************************************/

#pragma once

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <ctime>

using namespace std;

// The represetation for a feature and its value, init with '-1'
class FeaValNode
{
public:
	int iFeatureId;
	double dValue;

	FeaValNode (void);
	~FeaValNode (void);
};

// The represetation for a sample
class Sample
{
public:
	// the class index for a sample: 0-1 value, init with '-1'
	int iClass;
	vector<FeaValNode> FeaValNodeVec;

	Sample (void);
	~Sample (void);
};

// the minimal float number for smoothing for scaling the input samples
#define SMOOTHFATOR 1e-100

// The logistic regression classifier for MULTI-classes
class LogisticRegression
{
public:
	LogisticRegression(void);
	~LogisticRegression(void);

	// scale all of the sample values and put the result into txt
	bool ScaleAllSampleValTxt (const char * sFileIn, int iFeatureNum, const char * sFileOut);
	// train by SGD on the sample file
	bool TrainSGDOnSampleFile (
				const char * sFileName, int iClassNum, int iFeatureNum,		// about the samples
				double dLearningRate,										// about the learning 
				int iMaxLoop, double dMinImproveRatio						// about the stop criteria
				);
	// train by SGD on the sample file, decreasing dLearningRate during loop
	bool TrainSGDOnSampleFileEx (
				const char * sFileName, int iClassNum, int iFeatureNum,		// about the samples
				double dLearningRate,										// about the learning 
				int iMaxLoop, double dMinImproveRatio						// about the stop criteria
				);
	// train by SGD on the sample file, load all of the sample once a time and choose it randomly to train
	bool TrainSGDOnSampleFileEx2 (
				const char * sFileName, int iClassNum, int iFeatureNum,		// about the samples
				double dLearningRate,										// about the learning 
				int iMaxLoop, double dMinImproveRatio						// about the stop criteria
				);
	// save the model to txt file: the theta matrix with its size
	bool SaveLRModelTxt (const char * sFileName);
	// load the model from txt file: the theta matrix with its size
	bool LoadLRModelTxt (const char * sFileName);
	// load the samples from file, predict by the LR model
	bool PredictOnSampleFile (const char * sFileIn, const char * sFileOut, const char * sFileLog);

	// just for test
	void Test (void);

private:
	// read a sample from a line, return false if fail
	bool ReadSampleFrmLine (string & sLine, Sample & theSample);
	// load all of the samples into sample vector, this is for scale samples
	bool LoadAllSamples (const char * sFileName, vector<Sample> & SampleVec);
	// initialize the theta matrix with iClassNum and iFeatureNum
	bool InitThetaMatrix (int iClassNum, int iFeatureNum);
	// calculate the model function output for iClassIndex by feature vector
	double CalcFuncOutByFeaVec (vector<FeaValNode> & FeaValNodeVec, int iClassIndex);
	// calculate the model function output for all the classes, and return the class index with max probability
	int CalcFuncOutByFeaVecForAllClass (vector<FeaValNode> & FeaValNodeVec, vector<double> & ClassProbVec);
	// calculate the gradient and update the theta matrix, it returns the cost
	double UpdateThetaMatrix (Sample & theSample, vector<double> & ClassProbVec, double dLearningRate);
	// predict the class for one single sample
	int PredictOneSample (Sample & theSample);

private:
	// the number of target class
	int iClassNum;
	// the number of feature
	int iFeatureNum;
	// the theta matrix of iMaxFeatureNum * (iClassNum - 1)
	// note: for binary class, we need only 1 vector of theta; for multi-class, 
	// iMaxFeatureNum * (iClassNum - 1) is always enough
	vector< vector<double> > ThetaMatrix;
};
