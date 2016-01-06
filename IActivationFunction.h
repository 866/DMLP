//
// Created by victor on 24.12.15.
//

#include "common.h"

#ifndef DYNAMICMLP_IACTIVATIONFUNCTION_H
#define DYNAMICMLP_IACTIVATIONFUNCTION_H


class IActivationFunction {
public:
    virtual float calc(const float_vec& vec)=0;
};

class SigmoidFunction : public IActivationFunction {
public:
    SigmoidFunction(float alpha=0.5, float factor=1) : alpha{alpha}, factor{factor} { }
    float calc(const float_vec &vec)
    {
        float sum = std::accumulate(vec.begin(), vec.end(), 0.0);
        return factor / (1 + std::exp(-2.0 * this->alpha * sum));
    }

private:
    float alpha, factor;

};


#endif //DYNAMICMLP_IACTIVATIONFUNCTION_H
