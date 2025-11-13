//
// Created by jayde on 11/12/2025.
//

#include <iostream>
#include <vector>

#include "FloatWrapper.h"
#include "LinearNeuron.h"

int main()
{
	FloatWrapper input1 = {1};
	FloatWrapper input2 = {0};
	std::vector<Neuron*> layer1 = {&input1, &input2};
	LinearNeuron output = {layer1};
	output.UpdateToCurrentCalculation(1);
	std::cout << output.GetCurrentValue();
	input1.UpdateMyValue(0);
	input2.UpdateMyValue(1);
	std::cout << output.GetCurrentValue();
	input1.UpdateMyValue(2);
	input2.UpdateMyValue(2);
	std::cout << output.GetCurrentValue();
}
