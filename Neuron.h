//
// Created by jayde on 11/12/2025.
//

#ifndef LINALG_ML_NEURON_H
#define LINALG_ML_NEURON_H
#include <vector>


class Neuron {
	protected:
		std::vector<Neuron*> lastLayer;
        std::vector<float> weights;
		float bias;
		float calcNum;
		float myValue;
		static float startingWeightsMin;
		static float startingWeightsMax;
    public:
		Neuron(std::vector<Neuron*> lastLayer);
        virtual float GetCurrentValue();
		virtual void UpdateToCurrentCalculation(const int& calcNum) = 0;
};


#endif //LINALG_ML_NEURON_H