//
// Created by victor on 24.12.15.
//

#include "Layer.h"

float_vec Layer::calcDeltaWeightsSum() const {
    float_vec result((*this)[0].size() - 1);
    for(int i=0; i < result.size(); ++i) {
        for(int j=0; j < this->size(); ++j)
            result[i] += (*this)[j][i] * (*this)[j].getDelta();
    }
    return result;
}

std::ostream &operator<<( std::ostream &output,
                          const Layer &layer ) {
    output << "Float type artificial layer\n";
    output << "Number of nodes: " << layer.size() << std::endl;
}

float_vec Layer::getOutput() const {
    float_vec output;
    for(auto node : (*this))
        output.push_back(node.getOutput());
    return output;
}

void Layer::forward(const float_vec & inputs) {
    for(Node & node : (*this))
        node.forward(inputs);
}

void Layer::backward(const float_vec & targetsOrDeltas, bool isOutputLayer) {
    auto inputIter = targetsOrDeltas.cbegin();
    auto endIter = targetsOrDeltas.cend();
    auto nodeIter = this->begin();
    for( ; inputIter < endIter; inputIter++, nodeIter++) {
        (*nodeIter).backward(*inputIter, isOutputLayer);
    }
}

void Layer::updateDeltaWeights(const float & alpha, const float & learningRate, const float_vec & inputs) {
    for(Node & node : (*this))
        node.updateDeltaWeights(alpha, learningRate, inputs);
}

void Layer::updateWeights() {
    for(Node & node : (*this))
        node.updateWeights();
}
