
#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <cub/cub.cuh>

#include <stdio.h> 

using namespace std;

#define NUM_STATES 3
#define NUM_CHARS  256
#define NUM_THREADS 512
#define NUM_LINES 324
#define NUM_BLOCKS 400

#define BUFFER_SIZE 25000000
#define INPUT_FILE "./input_file.csv"
#define CSV_FILE 1 // 1: csv file, 0: txt file


typedef std::chrono::high_resolution_clock Clock;

template <int states>
struct __align__(4) state_array{
    uint8_t v[states];

    __device__ state_array() {
        for(int i = 0; i < states; i++)
            v[i] = i;
    }

    __device__ void set_SA(int index, int x) {
       v[index] = x;
    }

};

typedef state_array<NUM_STATES> SA;

struct SA_op {
    __device__ SA operator()(SA &a, SA &b){
        SA c;
        for(int i = 0; i < NUM_STATES; i++) 
            c.v[i] = b.v[a.v[i]];
        
        return c;
    }
};

/*
	remove_empty_elements:

	This function transfers the data from the pre-allocated array into the correct sized output array in order to 
	remove all internal fragmentations.
*/

__global__
void remove_empty_elements (int** input, int* len_array, int total_lines, int* index, int* temp_base, 
                            int* offset_array,  int* output, int* output_line_num, int taxi_application) {

    __shared__ int s_line_num;
    __shared__ int base;

    int len;
    int line_num;

    //get the next line to compute
    if(threadIdx.x == 0) 
        s_line_num = atomicInc((unsigned int*) index, INT_MAX);
    __syncthreads();
    // all threads have the same line_num
    line_num =  s_line_num;

    while(line_num < total_lines) {

    	//get the length of the line
        len = len_array[line_num];

        //get the offset in order to save the output into the correct place
		if(threadIdx.x == 0)
			base = offset_array[line_num];
        __syncthreads();
        
        // for loop for the line that is longer than NUM_THREADS(number of threads)
        for(int loop = threadIdx.x; loop < len; loop += NUM_THREADS) {

        	//special flag for the second run of data_parser. (if taxi_app is on, then it inserts comma in front of every line)
        	if(!taxi_application) {
        		//if the current character is in the line, it copies the value into the output array
        		if(loop < len){
               		 output[base + loop] = (input[line_num])[loop];
        		}
        	}
        	else {
        		//if the current character is in the line, it copies the value into the output array
        		//also save the line_num for the taxi application
        		if(loop < len - 1 ){
        			output_line_num[base + loop + 1] = line_num;
        			output[base + loop + 1] = (input[line_num])[loop] + 2;
        		}
        	}
        }
        __syncthreads();

        if(threadIdx.x == 0) {
        	//speical flag for taxi_app that puts 0 in front of all lines and saves all line_num for each coordinate
        	if(taxi_application){
        		output[base] = 0;
                output_line_num[base] = line_num;
            }
            //free the input array(it is dynamically allocated in merge_scan function)
            free(input[line_num]);
            //grab the next line
            s_line_num = atomicInc((unsigned int*) index, INT_MAX);
        }
         __syncthreads();
         //update the line_num for all threads
        line_num =  s_line_num;
    }

}


__global__
void merge_scan (char* line, int* len_array, int* offset_array, int** output_array, 
                 int* index, int total_lines, int* num_commas_array, SA* d_SA_Table, int* total_num_commas, uint8_t* d_E, int taxi_application){


    typedef cub::BlockScan<SA, NUM_THREADS> BlockScan_exclusive_scan; // change name
    typedef cub::BlockScan<int, NUM_THREADS> BlockScan_exclusive_sum; //

    __shared__ typename BlockScan_exclusive_scan::TempStorage temp_storage;
    __shared__ typename BlockScan_exclusive_sum::TempStorage temp_storage2;
    __shared__ SA prev_value;
    __shared__ int prev_sum;
    __shared__ int s_line_num;
    __shared__ int s_temp_array_size;
    __shared__ int* s_temp_ptr;

    SA temp_prev_val;
    int temp_prev_sum;

    int len, offset;
    int line_num;
    int start_state;

    int* temp_output_array;
    int temp_array_size;

    //grab the next/new line to compute
    if(threadIdx.x == 0) {
        s_line_num = atomicInc((unsigned int*) index, INT_MAX);
    }
    __syncthreads();
    line_num = s_line_num;

    //if the current line is in the input file 
    while(line_num < total_lines ) {

        temp_array_size = NUM_THREADS;
        //dynamic memory allocation
        if(threadIdx.x == 0) {
            temp_output_array = (int*)malloc(sizeof(int) * temp_array_size);
            output_array[line_num] = temp_output_array;
        }


        len = len_array[line_num];
        offset = offset_array[line_num];

        //initialize starting values
        SA a;
        prev_value = a;

        prev_sum = 0;
        temp_prev_sum = 0;
        int loop;
        __syncthreads();

        //If the string is longer than NUM_THREADS
        for(int ph = 0; ph < len; ph += NUM_THREADS) {

            loop = threadIdx.x + ph;
            char c = 0;

            if(loop < len) {
                c = line[loop + offset ];
	            a = d_SA_Table[c];
            }
            __syncthreads();
            //Merge SAs (merge the data to make one final sequence)
            BlockScan_exclusive_scan(temp_storage).ExclusiveScan(a, a, prev_value, SA_op(), temp_prev_val);
            __syncthreads();
           
            start_state = prev_value.v[0];
            int state = a.v[start_state];
            int start = (int) d_E[(int) (NUM_CHARS * state + c)];
            int end;
            //Excusive sum operation to find the number of commas
            BlockScan_exclusive_sum(temp_storage2).ExclusiveSum(start, end, temp_prev_sum);

            //(if the array is full, then it doubles the size fo the array )

            if(prev_sum + temp_prev_sum > temp_array_size){
            	int new_sum = prev_sum + temp_prev_sum;
            	int* temp_ptr;
            	//make a new array with double size
            	if(threadIdx.x == 0) {
            		while(new_sum > temp_array_size) {
            			temp_array_size = temp_array_size * 2;
            		}
            		s_temp_array_size = temp_array_size;
            		s_temp_ptr = (int*)malloc(sizeof(int) * temp_array_size);
            	}
            	//all threads have the same ptr and the array size
            	__syncthreads();
            	temp_array_size = s_temp_array_size;
            	temp_ptr = s_temp_ptr;
            	//copy the data 
            	for(int j = 0; j < (int)ceilf((float) (prev_sum) / NUM_THREADS); j++) {
            		int idx = threadIdx.x + j * NUM_THREADS;
            		if(idx < prev_sum) {
            			temp_ptr[idx] = output_array[line_num][idx];
            		}
            	}
            	__syncthreads();

            	//free the old array
            	if(threadIdx.x == 0) {
            		free(output_array[line_num]);
                    output_array[line_num] = temp_ptr;
            	}	
            }



            __syncthreads();

            //save the data (comma_index)
            if(start == 1 && loop < len) {
                (output_array[line_num])[end + prev_sum] = loop;
            }

            //save the end values for the next iteration
            if(threadIdx.x == 0) {
            	prev_value = temp_prev_val;
            	prev_sum += temp_prev_sum;
            }

            __syncthreads();

                    
        }

        //if the last thread saves the number of commas in the line
        if(loop == len - 1) {
        	//for the taxi app, it stores one more space (in front of every line) - this will be filled in remove_empty_elements kernel function
        	if(taxi_application)
				prev_sum++;
            num_commas_array[line_num] = prev_sum;
            //atomic operation to track total number of commas in the input file to properly allocated the space for the output array
            int temp = atomicAdd(total_num_commas, prev_sum);
        }

        //to get the next line
        if(threadIdx.x == 0) 
            s_line_num = atomicInc((unsigned int*) index, INT_MAX);
         __syncthreads();
        line_num =  s_line_num;
    }

}

/*
	This is just a function that calles ExclusiveSum function.
*/

__global__
void output_sort(int* input, int len, int* output) {
    typedef cub::BlockScan<int, NUM_THREADS> BlockScan; 
    __shared__ typename BlockScan::TempStorage temp_storage;
    __shared__ int prev_sum;

    int temp_prev_sum = 0;
    prev_sum = 0;

    //if the len is longer than the number of threads
    for(int ph = 0; ph < (int)ceilf((float) (len) / NUM_THREADS); ph ++) {
    	int loop = threadIdx.x + ph * NUM_THREADS;
    	temp_prev_sum = prev_sum;

	    int start = input[loop];
	    int end;
	    BlockScan(temp_storage).ExclusiveSum(start, end, temp_prev_sum);
	    
	    if(loop < len)
	    	output[loop] = end + prev_sum;
	    __syncthreads();
	    //save the last value as well (the output array has one more element than the input array because the first element is 0)
        if(loop == len - 1)
            output[loop + 1] = temp_prev_sum + prev_sum;
        __syncthreads();
        
	    if (threadIdx.x == 0)
	    	prev_sum += temp_prev_sum;
	    __syncthreads();

    }

    __syncthreads();


}


/*
	This function computes the length and the start index for the polyline (last flied) and the label (taxi ID)
*/

__global__
void polyline_coords (char* buffer, int* len_array, int* offset_array, int* comma_offset_array, int* comma_array,
                    int* output_len_array, int* output_offset_array, int* label_len_array, int* label_offset_array, int total_lines){

		// each threads gets one line
        int loop = threadIdx.x + blockIdx.x * blockDim.x;
        if(loop < total_lines) {
            int offset = offset_array[loop];
            int comma_offset = comma_offset_array[loop];
            int len = len_array[loop];

            //find the start and end of the polyline (last field)
            int start_idx = offset + comma_array[comma_offset + 7] + 3; 
            int end_idx = offset + len - 2;

            //save the length and start_idx of the polyline
            output_len_array[loop] = end_idx - start_idx;
            output_offset_array[loop] = start_idx; // -1 for the first index

            //find the start and the end of the label
            int label_start_idx = offset + 1;
            int label_end_idx = offset + comma_array[comma_offset] - 1;
            //find the length of the label
            int label_len = label_end_idx - label_start_idx;

            //save the length and start_idx of the label
            label_len_array[loop] = label_len;
            label_offset_array[loop] = label_start_idx;

        }
    
}
/*
	This function computes the length of the output line for the each coordinate, whcih is the sum of the lenght of label and the lenght of 
	the coordinate. 
*/

__global__
void coord_len_offset(  char* buffer, int* len_array, int* offset_array, int* line_idx_array, int* p_array, int* p_offset_array, int* p_comma_offset_array, int total_num, int garbage_char,
                        int* c_len_array, int* label_len_array) {

		//each threads computes one coordinate
        int coord_num = threadIdx.x + blockIdx.x * blockDim.x;
        int len;

        if(coord_num < total_num) {
        	//get the line number of the coordinate
            int line_num = line_idx_array[coord_num];
            int comma_off = p_comma_offset_array[line_num + 1];
            //get the index of the starting comma
            int cur = p_array[coord_num];

            //if the coordinate is the last coordinate of the line
            if(coord_num == comma_off - 1){
                len = len_array[line_num] - (p_offset_array[line_num] - offset_array[line_num]) - cur - garbage_char - CSV_FILE;

            }
            else {
            	//get the index of the ending comma
                int next = p_array[coord_num + 1];
                //garbage_char is the number of characters that are meaningless (e.g. "")
                len = next - cur - garbage_char;
            }   
            int label_len = label_len_array[line_num];
            //length of coordinate = len of coordinate + len of label
            c_len_array[coord_num] = (len + label_len);

        }

}


__global__
void switch_xy(char* buffer, int* line_idx_array,int* polyline_array, int* p_offset_array, int* c_len_array, int* c_offset_array,
                char* switched_array, int total, int total2, int* label_len_array, int* label_offset_array){

    // shared memory variables 
    __shared__ int comma_idx;
    __shared__ int line_num;
    __shared__ int label_len;
    __shared__ int label_offset;

    // set current line number, Taxi ID length, and offset to Taxi ID in buffer
    int block_num = blockIdx.x;         // current coordinate index within polyline 
    if(threadIdx.x == 0){
        line_num = line_idx_array[block_num];
        label_len = label_len_array[line_num];
        label_offset = label_offset_array[line_num];
    }
    __syncthreads();

    int cur = polyline_array[block_num];                // sets current coordinate comma in polyline
    int len = c_len_array[block_num] - label_len;       // subtracts taxi id length from coordinate length
    int offset = c_offset_array[block_num];             // offset to find a particular coordinate
    long start_idx = cur + p_offset_array[line_num];    // index to coordinate inside buffer

    // save taxi id in output array
    if(threadIdx.x < label_len) {
        switched_array[offset + threadIdx.x] = buffer[threadIdx.x + label_offset];
    }
    // calculate the position of coordinates in the polyline and switch latitude and longitude
    else if(threadIdx.x < len + label_len) {
        // set each thread to look at one character 
        int coord_idx = threadIdx.x - label_len;
        // if comma is found, save comma index in shared memory
        if(buffer[start_idx + coord_idx] == ',')
            comma_idx = coord_idx;
        __syncthreads();
        // calculate position relative to the comma index
        int position = coord_idx - comma_idx;
        // check cases to switch x and y positions
        if((coord_idx == 0) || (coord_idx == len - 1) ){ // edge cases
            switched_array[offset + coord_idx + label_len ] = buffer[start_idx + coord_idx];
        }
        else if(position == 1) { // if character is space after the comma 
            switched_array[offset + len - coord_idx + label_len] = buffer[start_idx + coord_idx];
        }
        else if(position == 0){ // if character is comma
            switched_array[offset + len - 2 - coord_idx + label_len] = buffer[start_idx + coord_idx];
        }
        else if(position > 0){ // if character is after comma
            switched_array[offset + position - 1 + label_len] = buffer[start_idx + coord_idx];
        }
        else{ // if character is before comma
            switched_array[offset + len - 1 - abs(position) + label_len] = buffer[start_idx + coord_idx];
        }

    }
    
}

__global__
void clear_array (int* input_array, int len) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < len) {
        input_array[idx] = 0;
    }

}


//CPU functions

int     D[NUM_STATES][NUM_CHARS];
uint8_t E[NUM_STATES][NUM_CHARS];
SA 		SA_Table[NUM_CHARS];

void add_transition (int state, uint8_t input, int next_state) 
{
    D[state][input] = next_state;
}

void add_default_transition(int state, int next_state) 
{
    for (int i = 0; i < NUM_CHARS; i++) 
        D[state][i] = next_state;
}

void add_emission(int state, uint8_t input, uint8_t value) 
{
    E[state][input] = value;
}

void add_default_emission(int state, uint8_t value) 
{
    for (int i = 0; i < NUM_CHARS; i++) 
        E[state][i] = value;
}

void SA_generate () {
	for (int i = 0; i < NUM_CHARS; i++) {
		for(int j = 0; j < NUM_STATES; j++) {
			(SA_Table[i]).v[j] = D[j][i];
		}
	}
}


void Dtable_generate() 
{
    for (int i = 0; i < NUM_STATES; i++) 
        add_default_transition(i ,i);
    
    add_default_transition(2 , 1);
   // add_default_transition(3 , 0);

    add_transition(0, '[', 1);
    add_transition(1, '\\', 2);
    add_transition(1, ']', 0);
 //   add_transition(0, '\\', 3);
}

void Etable_generate() 
{
    for(int i = 0; i < NUM_STATES; i++) 
        add_default_emission(i, 0);
    
    add_emission(0, ',', 1);
}

int main() {
    // generate transition tables for constant lookup time
    Dtable_generate();
    Etable_generate();
    SA_generate();
    // allocate device memory for state tables
    SA* d_SA_Table;
    cudaMalloc((SA**) &d_SA_Table, NUM_CHARS * sizeof(SA));

    uint8_t* d_E;
    cudaMalloc((uint8_t**) &d_E, NUM_STATES * NUM_CHARS * sizeof(uint8_t));

    // copy state tables from host to device
    cudaMemcpy(d_E, E, NUM_STATES * NUM_CHARS * sizeof(uint8_t), cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_SA_Table, SA_Table, NUM_CHARS * sizeof(SA), cudaMemcpyHostToDevice);

    // open input file
    std::ifstream is(INPUT_FILE);

    // get length of file:
    is.seekg (0, std::ios::end);
    long length = is.tellg();
    is.seekg (0, std::ios::beg);
    // if length of file is greater than buffer, send error message
    if(length > BUFFER_SIZE){
        cout<<"Error: File is too large to be read to buffer"<<endl;
    }
    else{
        string line; 
        long line_length;       // holds length of line being read
        long line_count = 0;    // counts number of lines read
        long char_offset = 0;   // offset of line into buffer
        int total_num_commas;   // number of commas 

        // allocate memory for arrays 
        char* buffer = new char [BUFFER_SIZE];
        int* len_array = new int[NUM_LINES];
        int* offset_array = new int[NUM_LINES + 1];
       
        // initialize offset array
        offset_array[0] = 0;   
        // read line by line from input file 
        while (getline(is, line)){
            // keep track of line length 
            line_length = line.size();

            // keep track of lengths of each line
            len_array[line_count] = line_length;

            // update offset from start of file
            char_offset += line_length + 1;
            offset_array[line_count + 1] = char_offset;

            // increment line index
            line_count++;

        }
        is.close();
        // reopen file stream
        std::ifstream is(INPUT_FILE);

        // read data as a block:
        is.read (buffer,length);

        // close filestream
        is.close();

        //Memory allocation for kernel functions

        int buffer_len = offset_array[line_count];
    
        int** d_output_array;
        cudaMalloc((int**)&d_output_array, line_count * sizeof(int*));

        char* d_buffer;
        cudaMalloc((char**) &d_buffer, buffer_len * sizeof(char));

        int* d_len_array;
        cudaMalloc((int**) &d_len_array, line_count * sizeof(int));

        int* d_offset_array;
        cudaMalloc((int**) &d_offset_array, line_count * sizeof(int));


        int* d_num_commas;
        cudaMalloc((int**) &d_num_commas, line_count * sizeof(int));


        int* d_comma_offset_array;
        cudaMalloc((int**)&d_comma_offset_array, (line_count + 1) * sizeof(int));

        int* d_stack;
        cudaMalloc((int**) &d_stack, sizeof(int));

        int* d_temp_base;
        cudaMalloc((int**) &d_temp_base, sizeof(int));

        int* d_total_num_commas;
        cudaMalloc((int**) &d_total_num_commas, sizeof(int));


        int temp = 0;

        // copies host memory to device memory after allocation 
        cudaMemcpy(d_buffer, buffer, buffer_len * sizeof(char), cudaMemcpyHostToDevice);     
        cudaMemcpy(d_len_array, len_array, line_count * sizeof(int), cudaMemcpyHostToDevice);     
        cudaMemcpy(d_offset_array, offset_array, line_count * sizeof(int), cudaMemcpyHostToDevice);    
        cudaMemcpy(d_stack, &temp, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_temp_base, &temp, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_total_num_commas, &temp, sizeof(int), cudaMemcpyHostToDevice);
        // synchronizes gpu functions with cpu 
     //   cudaDeviceSynchronize();    
        // grid and block dimensions to launch kernel function
        dim3 dimGrid(NUM_BLOCKS,1,1);
        dim3 dimBlock(NUM_THREADS,1,1);
        // function call to locate commas in input line
        merge_scan<<<dimGrid, dimBlock>>>(d_buffer, d_len_array, d_offset_array, d_output_array, d_stack, line_count, d_num_commas, d_SA_Table, d_total_num_commas, d_E, 0);
        // waits until all blocks in gpu finish running
       // cudaDeviceSynchronize();
        // sorts output 
        output_sort<<<1, NUM_THREADS>>> (d_num_commas, line_count, d_comma_offset_array);
        // copies number of commas to host memory
        cudaMemcpy(&total_num_commas, d_total_num_commas, sizeof(int), cudaMemcpyDeviceToHost);

        // creates an array to hold commas in device memory
        int* d_final_array;
        cudaMalloc((int**) &d_final_array, total_num_commas * sizeof(int));
      
        // makes a temporary d_stack
        cudaMemcpy(d_stack, &temp, sizeof(int), cudaMemcpyHostToDevice);
        // synchronizes gpu functions with cpu 
        //cudaDeviceSynchronize();
        // launches kernel function to clear unnecessary information
        remove_empty_elements<<<dimGrid, dimBlock>>> (d_output_array, d_num_commas, line_count, d_stack, d_temp_base, d_comma_offset_array, d_final_array, d_final_array /* temp array */, 0);
        // synchronizes gpu functions with cpu 
       // cudaDeviceSynchronize();
      
        // allocates memory 
        int* d_polyline_len_array;
        cudaMalloc((int**) &d_polyline_len_array, line_count * sizeof(int));

        int* d_polyline_offset_array;
        cudaMalloc((int**) &d_polyline_offset_array, (line_count + 1) * sizeof(int));

	    int* d_label_len_array;
        cudaMalloc((int**) &d_label_len_array, line_count * sizeof(int));

        int* d_label_offset_array;
        cudaMalloc((int**) &d_label_offset_array, line_count * sizeof(int));

        int* d_polyline_num_commas;
        cudaMalloc((int**) &d_polyline_num_commas, line_count * sizeof(int));

        // grid dimensions for polyline kernel function 
        dim3 dimGridPoly(ceil((float)line_count/NUM_THREADS),1,1);

        //cudaDeviceSynchronize();
        // launches function to calculate starting index of position of polyline, length of polyline, index of label, length of label
        polyline_coords<<<dimGridPoly, dimBlock>>>(d_buffer, d_len_array, d_offset_array, d_comma_offset_array, d_final_array, 
                d_polyline_len_array, d_polyline_offset_array, d_label_len_array, d_label_offset_array, line_count);

        //cudaDeviceSynchronize();
        //reset temp values for kernel functions
        cudaMemcpy(d_stack, &temp, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_temp_base, &temp, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_total_num_commas, &temp, sizeof(int), cudaMemcpyHostToDevice);

        //cudaDeviceSynchronize();

        // finds the comma indices within polyline
        merge_scan<<<dimGrid, dimBlock>>>(d_buffer, d_polyline_len_array, d_polyline_offset_array, d_output_array, d_stack, line_count, d_polyline_num_commas, d_SA_Table, d_total_num_commas, d_E, 1);

        //cudaDeviceSynchronize();

        // output to output sort function 
        int* d_polyline_comma_offset_array2;
        cudaMalloc((int**) &d_polyline_comma_offset_array2, sizeof(int) * (line_count + 1));

        // sorts the output to have the array hold the correct index
        output_sort<<<1, NUM_THREADS>>> (d_polyline_num_commas, line_count, d_polyline_comma_offset_array2);

        //cudaDeviceSynchronize();
        // finds the total number of commas in polyline
        int polyline_total_num_commas;
        cudaMemcpy(&polyline_total_num_commas, d_total_num_commas, sizeof(int), cudaMemcpyDeviceToHost);
        // uses number of commas in polyline to allocate output array in device memory
        int* d_polyline_array;
        cudaMalloc((int**) &d_polyline_array, polyline_total_num_commas * sizeof(int));
       
        // copies temp values to variable
        cudaMemcpy(d_stack, &temp, sizeof(int), cudaMemcpyHostToDevice);

//        cudaDeviceSynchronize();

        // creates array to find line number using comma index in polyline
        int* d_line_num_array;
        cudaMalloc((int**) &d_line_num_array, sizeof(int) * polyline_total_num_commas);

        // clears unnecessary elements 
        remove_empty_elements<<<dimGrid, dimBlock>>> (d_output_array, d_polyline_num_commas, line_count, d_stack, d_temp_base, d_polyline_comma_offset_array2, d_polyline_array, d_line_num_array, 1);

       // cudaDeviceSynchronize();
       
        //switch_xy setup

        // allocates memory to hold necessary information about coordinates
        int* c_len_array = new int[polyline_total_num_commas];
        int* c_offset_array = new int[polyline_total_num_commas + 1];

        int* d_c_len_array;
        cudaMalloc((int**) &d_c_len_array, polyline_total_num_commas * sizeof(int));
        int* d_c_offset_array;
        cudaMalloc((int**) &d_c_offset_array, (polyline_total_num_commas + 1) * sizeof(int));

        //cudaDeviceSynchronize();
        // sets up grid and block to launch kernel function
        dim3 dimGridcoord(ceil((float)polyline_total_num_commas / NUM_THREADS), 1, 1);
        dim3 dimBlockcoord(NUM_THREADS, 1, 1);
        // finds the length of each coordinate pair in polyline
        coord_len_offset<<<dimGridcoord, dimBlockcoord>>>(d_buffer, d_len_array, d_offset_array, d_line_num_array, d_polyline_array, d_polyline_offset_array, d_polyline_comma_offset_array2, polyline_total_num_commas, 
                                                          2, d_c_len_array, d_label_len_array);
        //cudaDeviceSynchronize();
        cudaMemcpy(c_len_array, d_c_len_array, polyline_total_num_commas * sizeof(int), cudaMemcpyDeviceToHost);
        //cudaDeviceSynchronize();
        // sorts the output to hold correct indices
        output_sort<<<1, NUM_THREADS>>>(d_c_len_array, polyline_total_num_commas , d_c_offset_array);
        cudaMemcpy(c_offset_array, d_c_offset_array, (polyline_total_num_commas + 1) * sizeof(int), cudaMemcpyDeviceToHost);


        //switch_xy setup

        // finds the size to fit the coordinates of polyline
        int coord_size;
        cudaMemcpy(&coord_size, (int*) (d_c_offset_array + polyline_total_num_commas), sizeof(int), cudaMemcpyDeviceToHost);
        // creates output array to hold coordinates
        char* switched_array = new char[coord_size];
        // creates output array in device memory
        char* d_switched_array;
        cudaMalloc((int**) &d_switched_array, coord_size * sizeof(char));

        // sets up grid and block dimensions for kernel function
        dim3 coordGrid(polyline_total_num_commas,1,1);
        dim3 coordBlock(NUM_THREADS,1,1);

        //cudaDeviceSynchronize();
        // launches kernel function to switch x and y positions of each coordinate in polyline 
        switch_xy<<<coordGrid,coordBlock>>>(d_buffer, d_line_num_array, d_polyline_array, d_polyline_offset_array, d_c_len_array, d_c_offset_array,
                                            d_switched_array, coord_size, polyline_total_num_commas, d_label_len_array, d_label_offset_array);

        //cudaDeviceSynchronize();
        // move output array from device to host 
        cudaMemcpy(switched_array, d_switched_array, coord_size * sizeof(char), cudaMemcpyDeviceToHost);     

        //cudaDeviceSynchronize();
        // prints the coordinates of each line with taxi id and switched x and y coordinates
         for(int i = 0; i < polyline_total_num_commas; i++) {
            int c_len = c_len_array[i];
            int c_off = c_offset_array[i];
            for(int j =0; j < c_len; j++){
                printf("%c",switched_array[c_off + j]);
            }
            cout << endl;
          }

        // device memory
	    cudaFree(d_polyline_array);
        cudaFree(d_polyline_len_array);
        cudaFree(d_polyline_offset_array);
        cudaFree(d_polyline_num_commas);
        cudaFree(d_polyline_comma_offset_array2);

        cudaFree(d_output_array);
        cudaFree(d_buffer);
        cudaFree(d_len_array);
        cudaFree(d_offset_array);
        cudaFree(d_comma_offset_array);

        cudaFree(d_stack);
        cudaFree(d_temp_base);
        cudaFree(d_num_commas);

        cudaFree(d_line_num_array);
        cudaFree(d_switched_array);
        cudaFree(d_c_len_array);
        cudaFree(d_c_offset_array);

        cudaFree(d_label_len_array);
        cudaFree(d_label_offset_array);
        cudaFree(d_total_num_commas);

        cudaFree(d_SA_Table);
        cudaFree(d_E);
        cudaFree(d_final_array);

        // delete temporary buffers
        delete [] buffer;
        delete [] len_array;
        delete [] offset_array;

        delete [] switched_array;
        delete [] c_len_array;
        delete [] c_offset_array;

    }


    return 0;
}


