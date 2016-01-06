//
// Created by victor on 24.12.15.
//
#ifndef PROCESSINGUNIT
#define PROCESSINGUNIT

#include "common.h"

template <class T>
class ProcessingUnit : public std::vector<T> {

public:
    virtual float_vec getOutput() const=0;
    virtual void forward(const float_vec & inputs)=0;
};
#endif // PROCESSINGUNIT

