//
// Created by victor on 23.12.15.
//
#include "common.h"
#include "IActivationFunction.h"

#ifndef DYNAMICMLP_NODE_H
#define DYNAMICMLP_NODE_H


class Node : public float_vec {

public:

    Node(int size, IActivationFunction* _afunc) :
            float_vec(size + 1), afunc(_afunc), delta_w(size + 1), delta(0) {
        fillRandom();
    }

    Node(const float_vec & initvec, IActivationFunction* _afunc) :
            float_vec(initvec), afunc(_afunc), delta_w(initvec.size()+1), delta(0) {
        if (initvec.size() < 1)
            throw std::domain_error("Neuron cannot have zero number of inputs.");
    }

    void fillRandom();
    void reset();
    void forward(const float_vec & inputs);
    void backward(const float &formulaMember, bool isOutputNode);
    void updateDeltaWeights(const float & alpha, const float & learningRate, const float_vec & inputs);
    void updateWeights();
    void resetDeltaWeights();

    float getDelta() const { return this->delta; }
    float getOutput() const { return output; }

    friend std::ostream &operator<<( std::ostream & output,
                                     const Node & node );
private:
    IActivationFunction* afunc;
    float output, delta;
    float_vec delta_w;

};


#endif //DYNAMICMLP_NODE_H
