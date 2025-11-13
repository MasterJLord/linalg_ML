//
// Created by jayde on 11/12/2025.
//

#include "LinearNeuron.h"

LinearNeuron::LinearNeuron(std::vector<Neuron *> lastLayer) : Neuron(lastLayer)
{
}


void LinearNeuron::UpdateToCurrentCalculation(const int& calcNum)
{
	if (this->calcNum == calcNum)
	{
		return;
	}
	myValue = bias;
	for (int i = 0; i < lastLayer.size(); i++)
	{
		lastLayer[i]->UpdateToCurrentCalculation(calcNum);
		myValue += weights[i] * lastLayer[i]->GetCurrentValue();
	}
}