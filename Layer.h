//
// Created by victor on 24.12.15.
//

#ifndef DYNAMICMLP_LAYER_H
#define DYNAMICMLP_LAYER_H

#include "Node.h"

class Layer : public std::vector<Node> {
    /*Class that represents layer of network
      Consits of several neurons*/
public:
    Layer(int nodes, int inputs, IActivationFunction* afunc) : std::vector<Node>(nodes, Node(inputs, afunc)) {}
    float_vec calcDeltaWeightsSum() const; // returns vector of sum(weights*deltas)
    float_vec getOutput() const; //

    void forward(const float_vec & inputs);
    void backward(const float_vec & targetsOrDeltas, bool isOutputLayer);
    void updateDeltaWeights(const float & alpha, const float & learningRate, const float_vec & inputs);
    void updateWeights();
    friend std::ostream &operator<<( std::ostream &output,
                                     const Layer &layer );

};


#endif //DYNAMICMLP_LAYER_H
