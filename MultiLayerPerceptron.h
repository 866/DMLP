//
// Created by victor on 24.12.15.
//

#ifndef DYNAMICMLP_MULTILAYERPERCEPTRON_H
#define DYNAMICMLP_MULTILAYERPERCEPTRON_H

#include <memory>
#include "Layer.h"

class MultiLayerPerceptron : public std::vector<Layer> {
    /*Class that represents multilayer perceptron*/

public:
    MultiLayerPerceptron(const std::vector<int> & structure, float _alpha=0.4, float _learningRate=.02);
    MultiLayerPerceptron(const MultiLayerPerceptron & mlp);
    MultiLayerPerceptron(const std::string & filepath);

    bool isBackwarded() const { return backwarded; }
    bool isForwarded() const { return forwarded; }

    void showAllNeurons() const; // prints all neurons and their states on the screen
    void addNewLayer();
    void forward(const float_vec & input); // pushes inputs through network
    void backward(const float_vec & target); // makes backward propagation of error
    void resetAllNodes(); // resets all neurons weights
    void updateDeltaWeights(); // updates delta_W
    void updateWeights(); // updates weights of neurons
    void saveToFile(const std::string & filepath) const;

    float_vec getOutput() const; // returns output if network has been forwarded

    friend std::ostream &operator<<( std::ostream & output,
                                     const MultiLayerPerceptron & mlp ); // gives ability to print network description


private:
    std::unique_ptr<IActivationFunction> activeFunc;
    std::vector<int> _structure;
    float_vec lastInput;
    bool forwarded = false;
    bool backwarded = false;
    float alpha, learningRate;
};


#endif //DYNAMICMLP_MULTILAYERPERCEPTRON_H
