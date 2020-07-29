// NNDigits.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "D:/OpenNN/opennn/opennn.h"


using namespace std;

int main()
{


	OpenNN::NeuralNetwork neu_network(784, 15, 10);//sets up the NN with 748 input, 15 hidden layer and 10 output neurons

	OpenNN::ModelSelection model;

	OpenNN::Variables* variables_ptr;

	OpenNN::Matrix<double> input_matrix;

	cout << "Gonna set Data" << endl;


	input_matrix.load("C:/Users/Mehmet Akyokus/Desktop/Train/data_matrix.dat");

	//Load data to data_set

	OpenNN::DataSet data_set(input_matrix);

	//data_set.load_data();//loads the data *******if the set_data_file_name does not, tough it says it does on function def

	OpenNN::Instances* instances_pointer = data_set.get_instances_pointer();

	instances_pointer->split_random_indices(0.7, 0.15, 0.15);// Training %, validation % , testing % of the data


	cout << "Data Set and Loaded" << endl;

	//set the pointers of input data

	variables_ptr = data_set.get_variables_pointer();

	for (double i = 0; i < 794; i++) {			//Sets the use of data_set, the first 784 are the pixels of picture, last 10 is the label(i.e. 1 0 0 0 0 0 0 0 0 0 is equal to "0")
		if (i < 784) {
			variables_ptr->set_use(i, OpenNN::Variables::Input);
		}
		else {
			variables_ptr->set_use(i, OpenNN::Variables::Target);

		}
	}

	cout << "Variable use given" << endl;


	//Setting the loss function parameters
	OpenNN::LossIndex loss_index(&neu_network, &data_set);

	loss_index.set_error_type(OpenNN::LossIndex::MEAN_SQUARED_ERROR);

	loss_index.set_regularization_type(OpenNN::LossIndex::NEURAL_PARAMETERS_NORM);

	//Setting the training strategy parameters

	OpenNN::TrainingStrategy training_strat(&loss_index); //Initialize the training strat with the loss index 

	training_strat.set_main_type(OpenNN::TrainingStrategy::GRADIENT_DESCENT); // Sets the main algorithm used to minimize cost function

	OpenNN::GradientDescent* gdMethod_ptr = training_strat.get_gradient_descent_pointer();

	gdMethod_ptr->set_maximum_iterations_number(50);
	gdMethod_ptr->set_display_period(5);
	gdMethod_ptr->set_maximum_time(15000);


	cout << "Starting Training" << endl;


	training_strat.perform_training();


	OpenNN::TestingAnalysis testing_analysis(&neu_network, &data_set);

	OpenNN::Matrix<size_t> confusion = testing_analysis.calculate_confusion();

	cout << "Confusion matrix coming" << endl;


	confusion.print();


	return 0;
}

