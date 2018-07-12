#include <iostream>
#include <fstream>      
#include <vector>
#include <string>
#include <stdio.h> // JDB -- should not be needed
#include<chrono>


using namespace std;

#define num_data_type 10
#define INPUT_FILE "./input_file.csv"
//#define INPUT_FILE "./input/go_track_trackspoints.csv"


typedef std::chrono::high_resolution_clock Clock;

int     D[4][256];
uint8_t E[4][256];

void add_transition (int state, uint8_t input, int next_state) {
	D[state][input] = next_state;
}

void add_default_transition(int state, int next_state) {
	for (int i = 0; i < 256; i++) {
		D[(int) state][i] = next_state;
	}
}

void add_emission(int state, uint8_t input, uint8_t value) {
	E[state][input] = value;
}

void add_default_emission(int state, uint8_t value) {
	for (int i = 0; i < 256; i++) {
		E[state][i] = value;
	}
}

void Dtable_generate(void) {
	for (int i = 0; i < 2; i++) {
		add_default_transition(i ,i);
	}
	add_default_transition(2 , 1);
	add_default_transition(3, 0);

	add_transition(0, '[', 1);
	add_transition(0, '\\', 3);
	add_transition(1, '\\', 2);
	add_transition(1, ']', 0);
}

void Etable_generate(void) {
	for(int i = 0; i < 3; i++) {
		add_default_emission(i, 0);
	}
	add_emission(0, ',', 1);
}

void parsedata (string s, uint8_t* comma_indices) {

	//default states
	int state = 0;
	int emission = 0;

	for (int i = 0; i < s.length(); i++) {
		emission = E[state][s[i]];
		state = D[state][s[i]];

		comma_indices[i] = emission;
	}
}

const int max_length(){

	std::ifstream is(INPUT_FILE);   // open file
	string line;
	int length = 0; 

	while (getline(is, line)){
		if(length < line.length())
			length = line.length();
	}
	is.close();
	
	return length; 
}

void seq_scan(uint8_t* comma_indices, int num_char) {
	for(int i = 1; i < num_char; i++) {
		comma_indices[i] = comma_indices[i - 1] + comma_indices[i];
	}
}

int main() {

	std::ifstream is(INPUT_FILE);   // open file
	string line;

	//generate the tables
	Dtable_generate();
	Etable_generate();

	// initialize bit vector
	const int array_len = max_length();	
	uint8_t* comma_indices = new uint8_t[array_len];

	for (int i = 0; i < array_len; i++) {
		comma_indices[i] = 0;
	}

                auto t1 = Clock::now();

	while (getline(is, line)) { 


		parsedata(line, comma_indices); 
		seq_scan(comma_indices, array_len);
		
		int prev = 0; 
		for(int i = 0; i < array_len; i++){
			if(prev != comma_indices[i]) {
				//std::cout << i << " ";
			}
			prev = comma_indices[i];
			comma_indices[i] = 0;
		}

		//cout << endl;

	}
	is.close();

                auto t2 = Clock::now();

        cout << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " microseconds" << endl;

	delete [] comma_indices;
    return 0;
}

