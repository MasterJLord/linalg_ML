//
// Created by jayde on 11/12/2025.
//

#ifndef LINALG_ML_LINEARNEURON_H
#define LINALG_ML_LINEARNEURON_H
#include "Neuron.h"


class LinearNeuron : public Neuron
{
public:
	LinearNeuron(std::vector<Neuron*> lastLayer);
	virtual void UpdateToCurrentCalculation(const int& calcNum);

};


#endif //LINALG_ML_LINEARNEURON_H