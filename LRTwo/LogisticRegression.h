/***********************************************************************************
* Logistic Regression classifier version 0.02
* Implemented by Jinghui Xiao (xiaojinghui@gmail.com or xiaojinghui1978@qq.com)
* Last updated on 2014-1-11
***********************************************************************************/

#pragma once

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <algorithm>
#include <cmath>

using namespace std;

// The represetation for a feature and its value
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

// The logistic regression 
class LogisticRegression
{
public:
	LogisticRegression(void);
	~LogisticRegression(void);

	// train by SGD on the sample file
	bool TrainSGDOnSampleFile (
				const char * sFileName, int iMaxFeatureNum,		// about the samples
				double dLearningRate,							// about the learning 
				int iMaxLoop, double dMinImproveRatio			// about the stop criteria
				);
	// save the model to txt file: the theta vector with its size
	bool SaveLRModelTxt (const char * sFileName);
	// load the model from txt file: the theta vector with its size
	bool LoadLRModelTxt (const char * sFileName);
	// load the samples from file, predict by the LR model
	bool PredictOnSampleFile (const char * sFileIn, const char * sFileOut, const char * sFileLog);

	// just for test
	void Test (void);

private:
	// the theta vector for each dimension of feature
	vector<double> ThetaVec;

	// read a sample from a line, return false if fail
	bool ReadSampleFrmLine (string & sLine, Sample & theSample);
	// the Sigmoid function: f(x) = 1 / (1 + exp(-x)) = exp (x) / ( 1 + exp(x) )
	double Sigmoid (double x);
	// calculate the model function output by feature vector
	double CalcFuncOutByFeaVec (vector<FeaValNode> & FeaValNodeVec);
	// calculate the gradient and update the theta value, it requires that sorts the feature index ascendingly
	void UpdateThetaVec (Sample & theSample, double dY, double dLearningRate);
	// predict the class for one single sample
	int PredictOneSample (Sample & theSample);
};

