//
// Created by victor on 26.12.15.
//

#ifndef DYNAMICMLP_TRAINER_H
#define DYNAMICMLP_TRAINER_H

#include "MultiLayerPerceptron.h"


using Sample=std::pair<float_vec, float_vec>; // Pair that consists of input and target vectors
using Samples=std::vector<Sample>;

class SamplesHandler {
public:
    SamplesHandler(const std::string & filepath, float test_rate = .05);
    Samples* getTrainingSamples();
    Samples* getTestSamples();

private:
    std::unique_ptr<Samples> _training_samples, _test_samples;
};

class Trainer {

public:
    Trainer(Samples* _train_samples, Samples* _test_samples,
            ulong _stopIter=1e+10, double _epsilonRMSE = 1e-30);

    void runTraining(MultiLayerPerceptron* mlp, int batches = 1, int testIteration=20, int verboseIteration=-1);

private:
    void getRMSEandAccuracy(MultiLayerPerceptron* mlp, double & RMSE, float & accuracy) const;
    Samples *train_samples, *test_samples;
    ulong stopIter;
    double epsilonRMSE;
};


#endif //DYNAMICMLP_TRAINER_H
