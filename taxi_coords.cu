#include "dataparser.h"

#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h> 

using namespace std;

int main(){

    parse_data("./taxi_input.txt");

    // int temp_id_array[3] = {15,443,1668};
    // int temp_comma_array[3] = {47,475,1700};

    // int offset_idx = 0; 
    // int start_idx = 0;
    // int input_line_idx = 0; 
    // int end_idx = 0; 
    // int polyline_len = 0; 
    // int taxi_id_len;

    // // increments by 8 to find each polyline 
    // // for(int i = 0; i < 20; i+=8){
    // for(int i = 0; i < 3; i++){

    //     start_idx = temp_comma_array[i]; // start of polyline

    //     input_line_idx = i; 

    //     end_idx = len_array[input_line_idx] + offset_idx; // end of line 

    //     polyline_len = end_idx - start_idx;
    //     char coords[polyline_len];
    //     strncpy(coords, buffer + start_idx, polyline_len);

    //     //taxi_id_len = temp_id_array[(i*2)+1] - temp_id_array[i*2];
    //     taxi_id_len = 8;
    //     //cout<<"taxiid:"<<taxi_id_len<<endl;
    //     char id[taxi_id_len];
    //     strncpy(id, buffer + temp_id_array[i], taxi_id_len);
    //     cout<<"id: "<<id<<endl;

    //     int len;
    //     int str_idx = 0;
    //     for(int j = 0; j < strlen(coords); j++){
    //         if(coords[j] == ',' && coords[j-1] == ']'){
    //             len = (j-1) - str_idx;
    //             printf("%.*s\n", len-1, coords + str_idx+2);
    //             str_idx = j;
    //         }
                
    //     }
    //     offset_idx += end_idx;
    // }

    return 0;
}