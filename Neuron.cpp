//
// Created by jayde on 11/12/2025.
//

#include "Neuron.h"
#include <random>


Neuron::Neuron(std::vector<Neuron*> lastLayer)
{
	this->lastLayer = lastLayer;
	this->weights = new std::vector<float>();
	for (int i = 0; i < lastLayer.size(); i++)
	{
		this->weights.push_back(rand() * (startingWeightsMax - startingWeightsMin) + startingWeightsMin);
	}
	this->calcNum = -1;
}

float Neuron::GetCurrentValue()
{
	return myValue;
}
