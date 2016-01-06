//
// Created by victor on 23.12.15.
//

#include <ctime>
#include "Node.h"

void Node::fillRandom() {
    auto getRandom = [=] { return (static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX)) - .5; };
    std::generate(this->begin(), this->end(), getRandom);
}

std::ostream &operator<<( std::ostream & output,
                          const Node & node ) {
    output << "Float type artificial neuron:\n";
    output << "\tsize: " << node.size() << "\n\tweights: [ ";
    for(auto weight: node)
        output << weight << ' ';
    output << "]\n\tdelta_w:[ ";
    for(auto delta: node.delta_w)
        output << delta << ' ';
    output << ']'<< std::endl;
    output << "\toutput: " << node.getOutput() << std::endl;
    output << "\tdelta: " << node.getDelta() << std::endl;
    return output;
}

void Node::forward(const float_vec & inputs) {
    if (inputs.size() != this->size() - 1) {
        throw std::domain_error("Inputs and weights mismatch.");
    }
    float_vec temp;
    for(int i=0; i<inputs.size(); ++i)
        temp.push_back(inputs[i] * (*this)[i]);
    temp.push_back(this->back());
    output = afunc->calc(temp);
}

void Node::backward(const float & formulaMember, bool isOutputNode) {
    if (isOutputNode)
        delta = output * (1 - output) * (formulaMember - output);
    else
        delta = output * (1 - output) * formulaMember;
}

void Node::reset() {
    fillRandom();
    resetDeltaWeights();
}

void Node::resetDeltaWeights() {
    std::fill(delta_w.begin(), delta_w.end(), 0);
}

void Node::updateDeltaWeights(const float & alpha, const float & learningRate, const float_vec & inputs) {
    auto delta_wIter = this->delta_w.begin();
    auto delta_wEndIter = this->delta_w.end() - 1;
    auto inputsIter = inputs.cbegin();
    for( ; delta_wIter < delta_wEndIter; ++delta_wIter, ++inputsIter)
        (*delta_wIter) = alpha * (*delta_wIter) +
                (1 - alpha) * learningRate * (*inputsIter) * delta;
    (*delta_wIter) = alpha * (*delta_wIter) + (1 - alpha) * learningRate * delta;
}

void Node::updateWeights() {
    std::transform(this->begin(), this->end(), delta_w.begin(),
                   this->begin(), std::plus<float>());
    resetDeltaWeights();
}
