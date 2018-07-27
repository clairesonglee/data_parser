#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <stdio.h> 

using namespace std;

#define INPUT_FILE "nla.csv"
#define NUM_STATES 3
#define NUM_THREADS 256

#define NUM_CHARS  256
#define NUM_LINES 3358602
#define NUM_BLOCKS 22

int main(){

	cout<<INPUT_FILE<<" "<<NUM_STATES<<" "<<NUM_THREADS<<" "<<endl;

	return 0; 
}
