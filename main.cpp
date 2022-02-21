/*******************************************************************************
* Copyright (c) 2015-2017
* School of Electrical, Computer and Energy Engineering, Arizona State University
* PI: Prof. Shimeng Yu
* All rights reserved.
*   
* This source code is part of NeuroSim - a device-circuit-algorithm framework to benchmark 
* neuro-inspired architectures with synaptic devices(e.g., SRAM and emerging non-volatile memory). 
* Copyright of the model is maintained by the developers, and the model is distributed under 
* the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License 
* http://creativecommons.org/licenses/by-nc/4.0/legalcode.
* The source code is free and you can redistribute and/or modify it
* by providing that the following conditions are met:
*   
*  1) Redistributions of source code must retain the above copyright notice,
*     this list of conditions and the following disclaimer. 
*   
*  2) Redistributions in binary form must reproduce the above copyright notice,
*     this list of conditions and the following disclaimer in the documentation
*     and/or other materials provided with the distribution.
*   
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
* 
* Developer list: 
*   Pai-Yu Chen     Email: pchen72 at asu dot edu 
*                     
*   Xiaochen Peng   Email: xpeng15 at asu dot edu
********************************************************************************/

#include <cstdio>
#include <iostream>
#include <ios>
#include <fstream>
#include <sstream>
#include <string>
#include <stdlib.h>
#include <random>
#include <vector>
#include "Cell.h"
#include "Array.h"
#include "formula.h"
#include "NeuroSim.h"
#include "Param.h"
#include "IO.h"
#include "Train.h"
#include "Test.h"
#include "Mapping.h"
#include "Definition.h"
#include "omp.h"
// #include <boost/date_time.hpp>

using namespace std;
std::vector< std::vector<double> > weight1totalE(param->nHide, std::vector<double>(param->nInput));
std::vector< std::vector<double> > weight2totalE(param->nOutput, std::vector<double>(param->nHide));
// std::vector< std::vector<double> > weight135to40(param->nHide, std::vector<double>(param->nInput));
// std::vector< std::vector<double> > weight235to40(param->nOutput, std::vector<double>(param->nHide));

extern double GAMA;

// Usage example: filePutContents("./yourfile.txt", "content", true);
void filePutContents(const std::string name, const double content, bool append = false) {
    std::ofstream outfile;
    if (append)
        outfile.open(name, std::ios_base::app);
    else
        outfile.open(name);
    outfile << content;
    outfile << endl;
}


int main() {
	gen.seed(0);

    RealDevice sample(0,0);
    double devLin = sample.linearity();
    cout << devLin << endl;
	
	/* Load in MNIST data */
	ReadTrainingDataFromFile("patch60000_train.txt", "label60000_train.txt");
	ReadTestingDataFromFile("patch10000_test.txt", "label10000_test.txt");

	/* Initialization of synaptic array from input to hidden layer */
	//arrayIH->Initialization<IdealDevice>();
	arrayIH->Initialization<RealDevice>(); 
	//arrayIH->Initialization<MeasuredDevice>();
	//arrayIH->Initialization<SRAM>(param->numWeightBit);
	//arrayIH->Initialization<DigitalNVM>(param->numWeightBit,true);
	//arrayIH->Initialization<HybridCell>(); // the 3T1C+2PCM cell
	//arrayIH->Initialization<_2T1F>();

	
	/* Initialization of synaptic array from hidden to output layer */
	//arrayHO->Initialization<IdealDevice>();
	arrayHO->Initialization<RealDevice>();
	//arrayHO->Initialization<MeasuredDevice>();
	//arrayHO->Initialization<SRAM>(param->numWeightBit);
	//arrayHO->Initialization<DigitalNVM>(param->numWeightBit,true);
	//arrayHO->Initialization<HybridCell>(); // the 3T1C+2PCM cell
	//arrayHO->Initialization<_2T1F>();


    omp_set_num_threads(16);
	/* Initialization of NeuroSim synaptic cores */
	param->relaxArrayCellWidth = 0;
	NeuroSimSubArrayInitialize(subArrayIH, arrayIH, inputParameterIH, techIH, cellIH);
	param->relaxArrayCellWidth = 1;
	NeuroSimSubArrayInitialize(subArrayHO, arrayHO, inputParameterHO, techHO, cellHO);
	/* Calculate synaptic core area */
	NeuroSimSubArrayArea(subArrayIH);
	NeuroSimSubArrayArea(subArrayHO);
	
	/* Calculate synaptic core standby leakage power */
	NeuroSimSubArrayLeakagePower(subArrayIH);
	NeuroSimSubArrayLeakagePower(subArrayHO);
	
	/* Initialize the neuron peripheries */
	NeuroSimNeuronInitialize(subArrayIH, inputParameterIH, techIH, cellIH, adderIH, muxIH, muxDecoderIH, dffIH, subtractorIH);
	NeuroSimNeuronInitialize(subArrayHO, inputParameterHO, techHO, cellHO, adderHO, muxHO, muxDecoderHO, dffHO, subtractorHO);
	/* Calculate the area and standby leakage power of neuron peripheries below subArrayIH */
	double heightNeuronIH, widthNeuronIH;
	NeuroSimNeuronArea(subArrayIH, adderIH, muxIH, muxDecoderIH, dffIH, subtractorIH, &heightNeuronIH, &widthNeuronIH);
	double leakageNeuronIH = NeuroSimNeuronLeakagePower(subArrayIH, adderIH, muxIH, muxDecoderIH, dffIH, subtractorIH);
	/* Calculate the area and standby leakage power of neuron peripheries below subArrayHO */
	double heightNeuronHO, widthNeuronHO;
	NeuroSimNeuronArea(subArrayHO, adderHO, muxHO, muxDecoderHO, dffHO, subtractorHO, &heightNeuronHO, &widthNeuronHO);
	double leakageNeuronHO = NeuroSimNeuronLeakagePower(subArrayHO, adderHO, muxHO, muxDecoderHO, dffHO, subtractorHO);
	
	/* Print the area of synaptic core and neuron peripheries */
	double totalSubArrayArea = subArrayIH->usedArea + subArrayHO->usedArea;
	double totalNeuronAreaIH = adderIH.area + muxIH.area + muxDecoderIH.area + dffIH.area + subtractorIH.area;
	double totalNeuronAreaHO = adderHO.area + muxHO.area + muxDecoderHO.area + dffHO.area + subtractorHO.area;
	printf("Total SubArray (synaptic core) area=%.4e m^2\n", totalSubArrayArea);
	printf("Total Neuron (neuron peripheries) area=%.4e m^2\n", totalNeuronAreaIH + totalNeuronAreaHO);
	printf("Total area=%.4e m^2\n", totalSubArrayArea + totalNeuronAreaIH + totalNeuronAreaHO);

	/* Print the standby leakage power of synaptic core and neuron peripheries */
	printf("Leakage power of subArrayIH is : %.4e W\n", subArrayIH->leakage);
	printf("Leakage power of subArrayHO is : %.4e W\n", subArrayHO->leakage);
	printf("Leakage power of NeuronIH is : %.4e W\n", leakageNeuronIH);
	printf("Leakage power of NeuronHO is : %.4e W\n", leakageNeuronHO);
	printf("Total leakage power of subArray is : %.4e W\n", subArrayIH->leakage + subArrayHO->leakage);
	printf("Total leakage power of Neuron is : %.4e W\n", leakageNeuronIH + leakageNeuronHO);
    

	
    // boost::posix_time::ptime timeLocal = boost::posix_time::second_clock::local_time();
    // boost::posix_time::time_duration durObj = timeLocal.time_of_day();

	/* Initialize weights and map weights to conductances for hardware implementation */
	WeightInitialize();
	if (param->useHardwareInTraining)
    	WeightToConductance();
	srand(0);	// Pseudorandom number seed
	

	ofstream mywriteoutfile;
    string filename = "output_nonlin " + to_string(devLin) + "_gamma "+  to_string(GAMA) +".csv";
	mywriteoutfile.open(filename);

    std::string files1 = "weights1totalE_nonlin " + to_string(devLin) + "_gamma " + to_string(GAMA) + ".txt";
    std::string files2 = "weights2totalE_nonlin " + to_string(devLin) + "_gamma " + to_string(GAMA) + ".txt";
    std::ofstream file1(files1);
    std::ofstream file2(files2);

    for (int i=1; i<=param->totalNumEpochs/param->interNumEpochs; i++){

   
        for (int j = 0; j < param->nOutput; j++) {
            for (int k = 0; k < param->nHide; k++) {
                weight2totalE[j][k] = weight2totalE[j][k] + weight2[j][k];
            }
        }

        for (int j = 0; j < param->nHide; j++) {
            for (int k = 0; k < param->nInput; k++) {
                weight1totalE[j][k] = weight1totalE[j][k] + weight1[j][k];
            }
        }
        
        for (int j = 0; j < param->nHide; j++) {
            for (int k = 0; k < param->nInput; k++) {
                filePutContents(files1,weight1totalE[j][k],true);
            }
        }
        file1.close();
        
        for (int j = 0; j < param->nOutput; j++) {
            for (int k = 0; k < param->nHide; k++) {
                filePutContents(files2,weight2totalE[j][k],true);
            }
        }
        file2.close();

        // for (int j = 0; j < param->nHide; j++) {
        //     for (int k = 0; k < param->nInput; k++) {
        //         file1 << weight1totalE[j][k] << "\n";
        //     }
        // }
        // for (int j = 0; j < param->nOutput; j++) {
        //     for (int k = 0; k < param->nHide; k++) {
        //         file2 << weight2totalE[j][k] << "\n";
        //     }
        // }

		Train(param->numTrainImagesPerEpoch, param->interNumEpochs,param->optimization_type);
		if (!param->useHardwareInTraining && param->useHardwareInTestingFF) { WeightToConductance(); }
		Validate();
        if (HybridCell *temp = dynamic_cast<HybridCell*>(arrayIH->cell[0][0]))
            WeightTransfer();
        else if(_2T1F *temp = dynamic_cast<_2T1F*>(arrayIH->cell[0][0]))
            WeightTransfer_2T1F();
                
		mywriteoutfile << i*param->interNumEpochs << ", " << (double)correct/param->numMnistTestImages*100 << endl;
		
		printf("Accuracy at %d epochs is : %.2f%\n", i*param->interNumEpochs, (double)correct/param->numMnistTestImages*100);
		/* Here the performance metrics of subArray also includes that of neuron peripheries (see Train.cpp and Test.cpp) */
		printf("\tRead latency=%.4e s\n", subArrayIH->readLatency + subArrayHO->readLatency);
		printf("\tWrite latency=%.4e s\n", subArrayIH->writeLatency + subArrayHO->writeLatency);
		printf("\tRead energy=%.4e J\n", arrayIH->readEnergy + subArrayIH->readDynamicEnergy + arrayHO->readEnergy + subArrayHO->readDynamicEnergy);
		printf("\tWrite energy=%.4e J\n", arrayIH->writeEnergy + subArrayIH->writeDynamicEnergy + arrayHO->writeEnergy + subArrayHO->writeDynamicEnergy);
		if(HybridCell* temp = dynamic_cast<HybridCell*>(arrayIH->cell[0][0])){
            printf("\tTransfer latency=%.4e s\n", subArrayIH->transferLatency + subArrayHO->transferLatency);
            printf("\tTransfer latency=%.4e s\n", subArrayIH->transferLatency);	
            printf("\tTransfer energy=%.4e J\n", arrayIH->transferEnergy + subArrayIH->transferDynamicEnergy + arrayHO->transferEnergy + subArrayHO->transferDynamicEnergy);
        }
        else if(_2T1F* temp = dynamic_cast<_2T1F*>(arrayIH->cell[0][0])){
            printf("\tTransfer latency=%.4e s\n", subArrayIH->transferLatency);	
            printf("\tTransfer energy=%.4e J\n", arrayIH->transferEnergy + subArrayIH->transferDynamicEnergy + arrayHO->transferEnergy + subArrayHO->transferDynamicEnergy);
         }
        // printf("\tThe total weight update = %.4e\n", totalWeightUpdate);
        // printf("\tThe total pulse number = %.4e\n", totalNumPulse);
        //std::cout << weight11to5[1][1] + weight21to5[1][1] << std::endl;
	}
	// print the summary: 
	printf("\n");


    /*
    std::cout << "epoch 1 ~ 5 weights1" << std::endl;
    std::cout << std::endl;
    for (int j = 0; j < param->nHide; j++) {
        for (int k = 0; k < param->nInput; k++) {
            std::cout << weight11to5[j][k]/5 << "\n";
        }
        std::cout << std::endl;
    }
	std::cout << "epoch 1 ~ 5 weights2" << std::endl;
    std::cout << std::endl;
    for (int j = 0; j < param->nOutput; j++) {
        for (int k = 0; k < param->nHide; k++) {
            std::cout << weight21to5[j][k]/5 << "\n";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << "epoch 35 ~ 40 weights1" << std::endl;
    std::cout << std::endl;
    for (int j = 0; j < param->nHide; j++) {
        for (int k = 0; k < param->nInput; k++) {
            std::cout << weight135to40[j][k]/5 << "\n";
        }
        std::cout << std::endl;
    }
	std::cout << "epoch 35 ~ 40 weights2" << std::endl;
    std::cout << std::endl;
    for (int j = 0; j < param->nOutput; j++) {
        for (int k = 0; k < param->nHide; k++) {
            std::cout << weight235to40[j][k]/5 << "\n";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << "Finished!" << std::endl;
    */


    // std::string files1 = "weights1totalE_" + to_simple_string(durObj) + ".csv";
    // std::string files2 = "weights2totalE_" + to_simple_string(durObj) + ".csv";
    // std::string files3 = "weights1final_" + to_simple_string(durObj) + ".txt";
    // std::string files4 = "weights2final_" + to_simple_string(durObj) + ".txt";

    /*
    std::ofstream file1("weights1ini.txt");
    std::ofstream file2("weights2ini.txt");
    std::ofstream file3("weights1final.txt");
    std::ofstream file4("weights2final.txt");
    */
    // std::ofstream file1(files1);
    // std::ofstream file2(files2);
    // std::ofstream file3(files3);
    // std::ofstream file4(files4);

    // for (int j = 0; j < param->nHide; j++) {
    //     for (int k = 0; k < param->nInput; k++) {
    //         file1 << weight1totalE[j][k] << "\n";
    //     }
    // }
    // for (int j = 0; j < param->nOutput; j++) {
    //     for (int k = 0; k < param->nHide; k++) {
    //         file2 << weight2totalE[j][k] << "\n";
    //     }
    // }
    // for (int j = 0; j < param->nHide; j++) {
    //     for (int k = 0; k < param->nInput; k++) {
    //         file3 << weight135to40[j][k]/5 << "\n";
    //     }
    // }
    // for (int j = 0; j < param->nOutput; j++) {
    //     for (int k = 0; k < param->nHide; k++) {
    //         file4 << weight235to40[j][k]/5 << "\n";
    //     }
    // }
	return 0;
}


