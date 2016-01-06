#include <iostream>
#include <memory>
#include <utility>
#include "MultiLayerPerceptron.h"
#include "Trainer.h"

int main() {
        SamplesHandler sampleHandler("/home/victor/Programming/MLP_tasks/523Rcircle");
        //std::unique_ptr<MultiLayerPerceptron> mlp(new MultiLayerPerceptron({2,60,30,2}, 0.1, .02 ));
        std::unique_ptr<MultiLayerPerceptron> mlp(new MultiLayerPerceptron("/home/victor/Programming/MLP_tasks/SavedMLP/52Rcircle.mlp"));
        //std::unique_ptr<Trainer> trainer(new Trainer(sampleHandler.getTrainingSamples(), sampleHandler.getTestSamples(), 1000000));
        //trainer->runTraining(mlp.get(), 1, 20000, 20000);
        //mlp->saveToFile("/home/victor/Programming/MLP_tasks/SavedMLP/52Rcircle.mlp");
        std::cout << "Start forward\n";
        mlp->forward(float_vec{3,2});
        std::cout << mlp->getOutput()[0] <<' ' << mlp->getOutput()[1] << std::endl;
        mlp->forward(float_vec{3,3});
        std::cout << mlp->getOutput()[0] <<' ' << mlp->getOutput()[1] << std::endl;
        mlp->forward(float_vec{-3.1,3});
        std::cout << mlp->getOutput()[0] <<' ' << mlp->getOutput()[1] << std::endl;
        return 0;
}
