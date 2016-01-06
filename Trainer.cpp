//
// Created by victor on 26.12.15.
//

#include "Trainer.h"


Trainer::Trainer(Samples *_train_samples, Samples *_test_samples, ulong _stopIter, double _epsilonRMSE) {
    test_samples = _test_samples;
    train_samples = _train_samples;
    stopIter = _stopIter;
    epsilonRMSE = _epsilonRMSE;
}

void Trainer::runTraining(MultiLayerPerceptron *mlp, int batches, int testIteration, int verboseIteration) {

    if (batches > stopIter)
        throw std::domain_error("Number of batches shouldn't be greater than total number of iterations.");

    int iteration = 0;
    int sampleNum = train_samples->size();
    double lastRMSE = 0, currentRMSE = 0;
    float accuracy = 0;
    getRMSEandAccuracy(mlp, currentRMSE, accuracy);
    if (verboseIteration != -1) {
        std::cout << "Starting RMSE: " << currentRMSE
                  << " ACCURACY: " << accuracy << std::endl;
    }


    while(iteration < this->stopIter &&
          (epsilonRMSE < 0 || std::fabs(lastRMSE-currentRMSE) > epsilonRMSE)) {

        const Sample & sample = (*train_samples)[rand() % sampleNum];
        mlp->forward(sample.first);
        mlp->backward(sample.second);
        mlp->updateDeltaWeights();

        iteration++;

        if (iteration % batches == 0)
            mlp->updateWeights();

        if (iteration % testIteration == 0) {
            lastRMSE = currentRMSE;
            getRMSEandAccuracy(mlp, currentRMSE, accuracy);
        }

        if ((verboseIteration != -1) && (iteration % verboseIteration == 0)) {
            std::cout << "Iteration #" << iteration << " RMSE: " << currentRMSE << " ACCURACY: " << accuracy << std::endl;
        }

    }
    if (verboseIteration != -1) {
        std::cout << "Iteration #" << iteration << " RMSE: " << currentRMSE << " ACCURACY: " << accuracy << std::endl;
    }
}

void Trainer::getRMSEandAccuracy(MultiLayerPerceptron* mlp, double & RMSE, float & accuracy) const
{
    double error = 0;
    unsigned long int success = 0;
    for(const Sample & sample : (*test_samples) ) {
        mlp->forward(sample.first);
        const float_vec & target = sample.second;
        const float_vec & output = mlp->getOutput();
        const static uint vec_size = output.size();
        for( int i = 0; i < vec_size; ++i) {
            double diff = static_cast<double>(target[i] - output[i]);
            error += (diff*diff);
        }
        if (std::distance(target.begin(), std::max_element(target.begin(), target.begin()+target.size())) ==
                std::distance(output.begin(), std::max_element(output.begin(), output.begin()+output.size())))
            success++;
    }
    RMSE = std::sqrt(error / static_cast<double>(test_samples->size()));
    accuracy = static_cast<float>(success) / test_samples->size();
}

SamplesHandler::SamplesHandler(const std::string & filepath, float test_rate){
    std::ifstream sampleFile;
    sampleFile.open(filepath);
    try {
        int inputsNumber, outputsNumber;
        unsigned long int total;
        sampleFile >> total >> inputsNumber >> outputsNumber;
        unsigned long int testTotal = std::ceil(static_cast<unsigned long int>(total * test_rate));
        unsigned long int trainingTotal = total - testTotal;
        _training_samples = std::unique_ptr<Samples>(new Samples(trainingTotal));
        _test_samples = std::unique_ptr<Samples>(new Samples(testTotal));
        int i = 0;
        unsigned long int sampleID = 0;
        float_vec inputs(inputsNumber), outputs(outputsNumber);
        for( ; sampleID < trainingTotal; ++sampleID) {
            for(i = 0; i < inputsNumber; ++i) sampleFile >> inputs[i];
            for(i = 0; i < outputsNumber; ++i) sampleFile >> outputs[i];
            (*_training_samples)[sampleID] = std::make_pair(inputs, outputs);
        }
        for(int sampleID = 0; sampleID < testTotal; ++sampleID) {
            for(i = 0; i < inputsNumber; ++i) sampleFile >> inputs[i];
            for(i = 0; i < outputsNumber; ++i) sampleFile >> outputs[i];
            (*_test_samples)[sampleID] = std::make_pair(inputs, outputs);
        }
        sampleFile.close();
    }
    catch(...) {sampleFile.close();}
}

Samples *SamplesHandler::getTrainingSamples()
{
    return _training_samples.get();
}

Samples *SamplesHandler::getTestSamples()
{
    return _test_samples.get();
}
