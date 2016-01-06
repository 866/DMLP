//
// Created by victor on 24.12.15.
//

#include "MultiLayerPerceptron.h"

MultiLayerPerceptron::MultiLayerPerceptron(const std::vector<int> & structure,
                                           float _alpha, float _learningRate) {
    activeFunc = std::unique_ptr<IActivationFunction>(new SigmoidFunction);
    this->alpha = _alpha; this->learningRate = _learningRate;
    auto iter = structure.cbegin();
    auto endIter = structure.cend();
    int inputs = *iter;
    iter++;
    for(; iter < endIter; ++iter) {
        int outputs = *iter;
        this->push_back(Layer(outputs, inputs, activeFunc.get()));
        inputs = outputs;
    }
}

MultiLayerPerceptron::MultiLayerPerceptron(const MultiLayerPerceptron &mlp)
{

}

MultiLayerPerceptron::MultiLayerPerceptron(const std::string & filepath) {
    std::ifstream mlpFile;
    mlpFile.open(filepath);
    try {
        activeFunc = std::unique_ptr<IActivationFunction>(new SigmoidFunction);
        int layers, numberOfWeights;
        mlpFile >> alpha >> learningRate >> layers;
        mlpFile >> numberOfWeights;
        for(int i = 0, neurons; i < layers; ++i) {
            mlpFile >> neurons;
            this->push_back(Layer(neurons, numberOfWeights, activeFunc.get()));
            numberOfWeights = neurons;
        }
        int i, j, k; float weight;
        while(mlpFile >> i >> j >> k >> weight)
            (*this)[i][j][k] = weight;
        mlpFile.close();
    }
    catch(...) {mlpFile.close();}
}

void MultiLayerPerceptron::resetAllNodes() {
    auto layerEndIter = this->end();
    auto layerIter=this->begin();
    for( ; layerIter < layerEndIter; ++layerIter){
        auto nodeEndIter = (*layerIter).end();
        for (auto nodeIter = (*layerIter).begin(); nodeIter < nodeEndIter; ++nodeIter)
            (*nodeIter).reset();
    }
    forwarded = false;
    backwarded = false;
}

std::ostream &operator<<(std::ostream &output, const MultiLayerPerceptron & mlp) {
    output << "MultiLayerPerceptron object.\n\tNumber of layers: " << mlp.size() << std::endl;
    output << "\tNumber of inputs: " << mlp[0][0].size() << std::endl;
    output << "\tNumber of outputs: " << mlp[mlp.size()-1].size() << std::endl;
    output << "\tNeurons in each layer: [ ";
    for(auto layer : mlp)
        output << layer.size() << ' ';
    output << "]\n";
    return output;
}

void MultiLayerPerceptron::forward(const float_vec & input) {
    lastInput = input;
    float_vec layerInput = lastInput;
    for(Layer & layer : (*this)) {
        layer.forward(layerInput);
        layerInput = layer.getOutput();
    }
    forwarded = true;
    backwarded = false;
}

void MultiLayerPerceptron::backward(const float_vec & target) {
    if (!forwarded)
        std::domain_error("Network cannot run in backward mode before forwarding.");

    auto layerIter = this->rbegin();
    auto layerEndIter = this->rend() - 1;
    layerIter->backward(target, true);
    auto deltaWeightsSum = (*layerIter).calcDeltaWeightsSum();
    layerIter++;
    for( ; layerIter < layerEndIter; ++layerIter) {
        (*layerIter).backward(deltaWeightsSum, false);
        deltaWeightsSum = (*layerIter).calcDeltaWeightsSum();
    }
    if (layerIter < layerEndIter+1)
        layerIter->backward(deltaWeightsSum, false);
    backwarded = true;
}

float_vec MultiLayerPerceptron::getOutput() const {
    if (!forwarded)
        std::domain_error("Output is not available. Please make forwarding.");

    return (*this)[this->size()-1].getOutput();
}

void MultiLayerPerceptron::updateDeltaWeights() {
    if (!backwarded)
        std::domain_error("Network cannot update weights without backward propagation.");
    auto layerIter = this->begin();
    auto layerEndIter = this->end();
    (*layerIter).updateDeltaWeights(alpha, learningRate, lastInput);
    layerIter++;
    for ( ; layerIter < layerEndIter; ++layerIter)
    {
        auto layerInput = (*(layerIter-1)).getOutput();
        (*layerIter).updateDeltaWeights(alpha, learningRate, layerInput);
    }
}

void MultiLayerPerceptron::updateWeights() {
    if (!backwarded)
        std::domain_error("Network cannot update weights without backward propagation.");

    for (Layer & layer : *this)
        layer.updateWeights();
}

void MultiLayerPerceptron::saveToFile(const std::string & filepath) const {
    std::ofstream mlpFile;
    mlpFile.open(filepath);
    try {
        int layers = this->size();
        std::vector<int> structure(layers + 1);
        structure[0] = ((*this)[0][0].size() - 1);
        mlpFile << alpha << ' ' << learningRate << ' ' << layers << '\n' << structure[0] << ' ';
        for(int i = 1; i < layers+1; i++) {
            structure[i] = (*this)[i-1].size();
            mlpFile << structure[i] << ' ';
        }
        mlpFile << '\n';
        for(int i = 1; i < structure.size(); ++i) // Layer iterator
            for(int j = 0; j < structure[i]; ++j) // Node iterator
                for(int k = 0; k < structure[i-1] + 1; ++k) // Weight iterator
                    mlpFile << i-1 << ' ' << j << ' ' << k << ' ' << (*this)[i-1][j][k] << '\n';

    }
    catch(...) {mlpFile.close();}
}

void MultiLayerPerceptron::showAllNeurons() const {
    std::cout << "Network neurons: " << std::endl;
    for(int i=0; i < this->size(); ++i)
        for(int j=0; j < (*this)[i].size(); ++j) {
            std::cout << "Neuron[" << i << "]" << "[" << j << "]: " << std::endl;
            std::cout << (*this)[i][j] << std::endl;
        }
}

void MultiLayerPerceptron::addNewLayer()
{

}
