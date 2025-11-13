//
// Created by jayde on 11/12/2025.
//

#ifndef LINALG_ML_NEURONWRAPPER_H
#define LINALG_ML_NEURONWRAPPER_H
#include "Neuron.h"


class FloatWrapper : public Neuron
{
	public:
		FloatWrapper(const float& myValue);
		void UpdateMyValue(const float& newValue);
		virtual void UpdateToCurrentCalculation(const int& calcNum);

};


#endif //LINALG_ML_NEURONWRAPPER_H