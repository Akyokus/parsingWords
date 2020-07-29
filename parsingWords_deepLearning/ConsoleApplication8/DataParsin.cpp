// This application converts the data from MNIST Data Set to a single .dat file that can be used by OpenNN

#include "stdafx.h"
#include "string.h"
#include <fstream>
#include <iostream>


#include <D:/OpenNN/opennn/opennn.h>
#include <D:/OpenNN/opennn/data_set.h>


using namespace std;


int ReverseInt(int i) {
	
	
	unsigned char ch1, ch2, ch3, ch4;

	ch1 = i & 255;

	ch2 = (i >> 8) & 255;

	ch3 = (i >> 16) & 255;

	ch4 = (i >> 24) & 255;

	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}


int doNothing() {
	return 6;
}


void read_Mnist(string filename, OpenNN::Matrix<double> &intputMatrix) {

	ifstream file(filename, ios::binary);

	string filename_label = "C:/Users/Mehmet Akyokus/Desktop/Train/train-labels.idx3-ubyte";
	ifstream file_label(filename_label, ios::binary);

	string filename_test_label = "C:/Users/Mehmet Akyokus/Desktop/Test/t10k-labels.idx1-ubyte";
	ifstream file_test_label(filename_test_label, ios::binary);

	string filename_test= "C:/Users/Mehmet Akyokus/Desktop/Test/t10k-images.idx3-ubyte";
	ifstream file_test(filename_test, ios::binary);

	if (file.is_open() && file_label.is_open()&& file_test_label.is_open() && file_test.is_open()){

		//Variables for TRAINING DATA and labels
		int magic_number = 0; //For this num ,the first 2 bytes are zero, 3rd byte specifies the type of data, 4 byte gives the dimension of data(1 for vector, 2 for matricies)

		int number_of_images = 0;

		int n_rows = 0;	// number of rows for input data

		int n_cols = 0; // number of cols for input data

		int label_index;//To turn (int) label to vector. For example, label = 9 is {0,0,0,0,0,0,0,0,0,1}

		int magic_number_label = 0; // These are for the label file of the training data

		int number_of_images_label = 0;

		int n_rows_label = 0;

		int n_cols_label = 0;


		//Variables for TEST DATA and labels
		int magic_number_test = 0;

		int number_of_images_test = 0;

		int n_rows_test = 0;

		int n_cols_test = 0;

		int label_index_test;

		int magic_number_label_test = 0;

		int number_of_images_label_test = 0;

		int n_rows_label_test = 0;

		int n_cols_label_test = 0;

		
		OpenNN::Vector<double> tempVec(794,0);//Every 28*28 picture is saved in this vector and then pushed to the Matrix
		OpenNN::Vector<double> resetVec(794, 0);// This vector is used the reset tempVec, there was no easy way to clear tempVec. Maybe add clear to vector.cpp from openNN

		
		//TRAINING
		
				
		//Label
		file_label.read((char*)&magic_number_label, sizeof(magic_number_label));
		
		magic_number_label = ReverseInt(magic_number_label); // Get magic number. This could be used to determine the type of data and how it is saved. But it is already known for this data set
		
		

		file_label.read((char*)&number_of_images_label, sizeof(number_of_images_label));
		
		number_of_images_label = ReverseInt(number_of_images_label);		
			

		//Data Set
		file.read((char*)&magic_number, sizeof(magic_number));

		magic_number = ReverseInt(magic_number);



		file.read((char*)&number_of_images, sizeof(number_of_images));

		number_of_images = ReverseInt(number_of_images);



		file.read((char*)&n_rows, sizeof(n_rows));

		n_rows = ReverseInt(n_rows);



		file.read((char*)&n_cols, sizeof(n_cols));

		n_cols = ReverseInt(n_cols);

		

		//TEST

		//Label
		file_test_label.read((char*)&magic_number_label_test, sizeof(magic_number_label_test));

		magic_number_label_test = ReverseInt(magic_number_label_test);



		file_test_label.read((char*)&number_of_images_label_test, sizeof(number_of_images_label_test));

		number_of_images_label_test = ReverseInt(number_of_images_label_test);

	

		file_test.read((char*)&magic_number_test, sizeof(magic_number_test));

		magic_number_test = ReverseInt(magic_number_test);



		file_test.read((char*)&number_of_images_test, sizeof(number_of_images_test));

		number_of_images_test = ReverseInt(number_of_images_test);



		file_test.read((char*)&n_rows_test, sizeof(n_rows_test));

		n_rows_test = ReverseInt(n_rows_test);



		file_test.read((char*)&n_cols_test, sizeof(n_cols_test));

		n_cols_test = ReverseInt(n_cols_test);



		unsigned char temp = 0;

		//TRAINING data is saved to the Matrix

		for (int i = 0; i < number_of_images; ++i){
			
			tempVec = resetVec;

			for (int r = 0; r < n_rows; ++r){

				for (int c = 0; c <(n_cols); ++c){ // Read characters and save them to tempVec

					temp = 0;						

					file.read((char*)&temp, sizeof(temp));

					tempVec[(r * 28) + c] = (double)temp;


				}
			
			
			}
			unsigned char temp_label = 0;

			file_label.read((char*)&temp_label, sizeof(temp_label)); // Get the label of the data saved to tempVec

			label_index = (double)temp_label + 784;  // get the index for the label. 

			tempVec[label_index] = 1; // Set the vector to 1 at the label_index location

			intputMatrix.set_row(i, tempVec); // Save tempVec to matrix
		}

		//TEST
		
		for (int i = 0; i < number_of_images_test; ++i) { // Same of above for the test data
			tempVec = resetVec;

			vector<double> tp_test;

			for (int r = 0; r < n_rows; ++r) {

				for (int c = 0; c <(n_cols); ++c) {

					temp = 0;

					file_test.read((char*)&temp, sizeof(temp));

					tempVec[(r * 28) + c] = (double)temp;


				}


			}
			unsigned char temp_label = 0;

			file_test_label.read((char*)&temp_label, sizeof(temp_label));

			label_index_test = (double)temp_label + 784;

			tempVec[label_index_test] = 1;

			intputMatrix.set_row(i+ number_of_images, tempVec); // tempVec is saved at "1+number_of_images" since it should come after training data
		}

		intputMatrix.print_preview();// prints the 1st, 2nd and last rows of the matrix
		
	}
}



int main()
	
{

	OpenNN::Matrix<double> m(70000,794,0); // Matrix that will be saved
	
	cout << " starting"<< endl;

	read_Mnist("C:/Users/Mehmet Akyokus/Desktop/Train/train-images.idx3-ubyte", m);
	
	//m.load("C:/Users/Mehmet Akyokus/Desktop/Train/data_matrix.dat");

	m.save("data_matrix.dat"); // Saves the matrix to .dat file. "Space" is used as separator. 

	
		 
	cout << " matrix row num " << m.get_rows_number() << endl;

	cout << " matrix cols num " << m.get_columns_number() << endl;

	m.print_preview();

    return 0;
}

