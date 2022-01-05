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

#ifndef CELL_H_
#define CELL_H_

#include <random>
#include <vector>

class Cell {
public:
	int x, y;	// Cell location: x (column) and y (row) start from index 0
	double heightInFeatureSize, widthInFeatureSize;	// Cell height/width in terms of feature size (F)
	double area;	// Cell area (m^2)
	virtual ~Cell() {}	// Add a virtual function to enable dynamic_cast
};

class eNVM: public Cell {
public:
	double readVoltage;	// On-chip read voltage (Vr) (V)
	double readPulseWidth;	// Read pulse width (s) (will be determined by ADC)
	double readEnergy;	// Dynamic variable for calculation of read energy (J)
	double writeVoltageLTP;	// Write voltage (V) for LTP or weight increase
	double writeVoltageLTD;	// Write voltage (V) for LTD or weight decrease
	double writePulseWidthLTP;	// Write pulse width (s) of LTP or weight increase
	double writePulseWidthLTD;	// Write pulse width (s) of LTD or weight decrease
	double writeEnergy;	// Dynamic variable for calculation of write energy (J)
	double conductance;	// Current conductance (S) (Dynamic variable) at on-chip Vr (different than the Vr in the reported measurement data)
	double conductancePrev;	// Previous conductance (S) (Dynamic variable) at on-chip Vr (different than the Vr in the reported measurement data)
	double maxConductance;	// Maximum cell conductance (S)
	double minConductance;	// Minimum cell conductance (S)
	double avgMaxConductance;   // Average maximum cell conductance (S)
	double avgMinConductance;   // Average minimum cell conductance (S)
	bool cmosAccess;	// True: Pseudo-crossbar (1T1R), false: cross-point
    bool isSTTMRAM; // if it is a STTMRAM device
    // modified above
	bool FeFET;			// True: FeFET structure (Pseudo-crossbar only, should be cmosAccess=1)
	double resistanceAccess;	// The resistance of transistor (Ohm) in Pseudo-crossbar array when turned ON
	bool nonlinearIV;	// Consider I-V nonlinearity or not (Currently this option is for cross-point array. It is hard to have this option in pseudo-crossbar since it has an access transistor and the transistor's resistance can be comparable to RRAM's resistance after considering the nonlinearity. In this case, we have to iteratively find both the resistance and Vw across RRAM.)
	bool readNoise;	// Consider read noise or not
	double sigmaReadNoise;	// Sigma of read noise in gaussian distribution
	double NL;	// Nonlinearity in write scheme (the current ratio between Vw and Vw/2), assuming for the LTP side
	std::normal_distribution<double> *gaussian_dist;	// Normal distribution object
	std::normal_distribution<double> *gaussian_dist2;	// Normal distribution object
	std::normal_distribution<double> *gaussian_dist3;	// Normal distribution object
	std::normal_distribution<double> *gaussian_dist4;	// Normal distribution object
	std::normal_distribution<double> *gaussian_dist5;	// Normal distribution object
	std::normal_distribution<double> *gaussian_dist_maxConductance;	// Normal distribution object
	std::normal_distribution<double> *gaussian_dist_minConductance;	// Normal distribution object
	/* Need the 4 variables below if nonlinearIV=true */
	double conductanceAtVwLTP;		// Conductance at the LTP write voltage
	double conductanceAtVwLTD;		// Conductance at the LTD write voltage
	double conductanceAtHalfVwLTP;	// Conductance at 1/2 LTP write voltage
	double conductanceAtHalfVwLTD;	// Conductance at 1/2 LTD write voltage
	bool conductanceRangeVar;	// Consider variation of conductance range or not
	double maxConductanceVar;	// Sigma of maxConductance variation (S)
	double minConductanceVar;	// Sigma of minConductance variation (S)
};

class SRAM: public Cell {
public:
	SRAM(int x, int y);
	int bit;	// Stored bit (1 or 0) (dynamic variable)
	int bitPrev;	// Previous bit
	double widthSRAMCellNMOS;	// Pull-down NMOS width in terms offeature size (F)
	double widthSRAMCellPMOS;	// Pull-up PMOS width in terms of feature size (F)
	double widthAccessCMOS;		// Access transistor width in terms of feature size (F)
	double minSenseVoltage;		// Minimum voltage difference (V) for sensing
	double readEnergy;			// Dynamic variable for calculation of read energy (J) 
	double writeEnergy;			// Dynamic variable for calculation of write energy (J)
	double readEnergySRAMCell;	// Read energy (J) per SRAM cell (currently not used, it is included in the peripheral circuits of SRAM array in NeuroSim)
	double writeEnergySRAMCell;	// Write energy (J) per SRAM cell (will be obtained from NeuroSim)
	double Read(){}	// Currently not used
	void Write(){}	// Currently not used
	bool parallelRead;  // parallel read or not
};

class AnalogNVM: public eNVM {
public:
	int maxNumLevelLTP;	// Maximum number of conductance states during LTP or weight increase
	int maxNumLevelLTD;	// Maximum number of conductance states during LTD or weight decrease
	int numPulse;   // Number of write pulses used in the most recent write operation (Positive number: LTP, Negative number: LTD) (dynamic variable)
	double writeLatencyLTP;	// Write latency of a cell during LTP or weight increase (different cells use different # write pulses, thus latency values are different). writeLatency will be calculated for each cell first, and then replaced by the maximum one in the batch write.
	double writeLatencyLTD;	// Write latency of a cell during LTD or weight decrease (different cells use different # write pulses, thus latency values are different). writeLatency will be calculated for each cell first, and then replaced by the maximum one in the batch write.
	bool FeFET;			// True: FeFET structure (Pseudo-crossbar only, should be cmosAccess=1)
	double gateCapFeFET;	// Gate Capacitance of FeFET (F)
	/* Non-identical write pulse scheme */
	bool nonIdenticalPulse;	// Use non-identical pulse scheme in weight update or not (put the parameter here due to the access from Train.cpp)
	double VinitLTP;    // Initial write voltage for LTP or weight increase (V)
	double VstepLTP;    // Write voltage step for LTP or weight increase (V)
	double VinitLTD;    // Initial write voltage for LTD or weight decrease (V)
	double VstepLTD;    // Write voltage step for LTD or weight decrease (V)
	double PWinitLTP;   // Initial write pulse width for LTP or weight increase (s)
	double PWstepLTP;   // Write pulse width for LTP or weight increase (s)
	double PWinitLTD;   // Initial write pulse width for LTD or weight decrease (s)
	double PWstepLTD;   // Write pulse width for LTD or weight decrease (s)
	double writeVoltageSquareSum;   // Sum of V^2 of non-identical pulses (for weight update energy calculation in subcircuits)

	virtual double Read(double voltage) = 0;
	virtual void Write(double deltaWeightNormalized, double weight, double minWeight, double maxWeight) = 0;
	double GetMaxReadCurrent(){
      if(cmosAccess && !FeFET)
          return readVoltage * 1/(1/avgMaxConductance+resistanceAccess);
      else 
          return readVoltage * avgMaxConductance;}
	double GetMinReadCurrent(){
      if(cmosAccess && !FeFET)
          return readVoltage * 1/(1/avgMinConductance+resistanceAccess);
      else
          return readVoltage * avgMinConductance;}
	void WriteEnergyCalculation(double wireCapCol);
};

class DigitalNVM: public eNVM {
public:
	DigitalNVM(int x, int y);
	int bit;	// Stored bit (1 or 0) (dynamic variable), for internel check only and not be used for read
	int bitPrev;	// Previous bit
	double refCurrent;	// Reference current for S/A
	double Read(double voltage);	// Return read current (A)
       // modified below
    bool isSTTMRAM;  // if it is STTMRAM, then, we can relax the cell area
    bool parallelRead; // if it is a parallel readout for STT-MRAM
	void Write(int bitNew, double wireCapCol);
};

class IdealDevice: public AnalogNVM {
public:
	IdealDevice(int x, int y);
	double Read(double voltage);	// Return read current (A)
	void Write(double deltaWeightNormalized, double weight, double minWeight, double maxWeight);
};

class RealDevice: public AnalogNVM {
public:
	bool nonlinearWrite;	// Consider weight update nonlinearity or not
	double xPulse;		// Conductance state in terms of the pulse number (doesn't need to be integer)
	double NL_LTP;		// LTP nonlinearity
	double NL_LTD;		// LTD nonlinearity
	double paramALTP;	// Parameter A for LTP nonlinearity
	double paramBLTP;	// Parameter B for LTP nonlinearity
	double paramALTD;	// Parameter A for LTD nonlinearity
	double paramBLTD;	// Parameter B for LTD nonlinearity
	double sigmaDtoD;	// Sigma of device-to-device variation on weight update nonliearity baseline
	double sigmaCtoC;	// Sigma of cycle-to-cycle variation on weight update

	RealDevice(int x, int y);
	double Read(double voltage);	// Return read current (A)
	void Write(double deltaWeightNormalized, double weight, double minWeight, double maxWeight);
};

class MeasuredDevice: public AnalogNVM {
public:
	bool nonlinearWrite;	// Consider weight update nonlinearity or not
	bool symLTPandLTD;	// True: use LTP conductance data for LTD
	double xPulse;		// Conductance state in terms of the pulse number (doesn't need to be integer)
	std::vector<double> dataConductanceLTP;	// LTP conductance data at different pulse number
	std::vector<double> dataConductanceLTD;	// LTD conductance data at different pulse number

	MeasuredDevice(int x, int y);
	double Read(double voltage);	// Return read current (A)
	void Write(double deltaWeightNormalized, double weight, double minWeight, double maxWeight);
};

// code added
class _3T1C : public Cell {
   public:
	double readVoltage;	    // On-chip read voltage (Vr) (V) for the LSB capacitor 
	double readPulseWidth;	// Read pulse width for the LSB capacitor (s) (will be determined by ADC)
	double readEnergy;	        // Dynamic variable for calculation of read energy (J)
	
    /* the write voltage and write current depends on the access transistor
        make sure they are consistent with each other*/
    double writeVoltageLTP;	 // Write voltage (V) for LTP or weight increase
	double writeVoltageLTD;	 // Write voltage (V) for LTD or weight decrease
    double writeCurrentLTP;  // Write current (A) for LTP or weight increase
    double writeCurrentLTD; // Write current (A) for LTP or weight increase

	double writePulseWidthLTP;	// Write pulse width (s) of LTP or weight increase
	double writePulseWidthLTD;	// Write pulse width (s) of LTD or weight decrease
	double writeEnergy;	            // Dynamic variable for calculation of write energy (J)
    double writeLatencyLTP;	// Write latency of a cell during LTP or weight increase (different cells use different # write pulses, thus latency values are different). writeLatency will be calculated for each cell first, and then replaced by the maximum one in the batch write.
	double writeLatencyLTD;	// Write latency of a cell during LTD or weight decrease (different cells use different # write pulses, thus latency values are different). writeLatency will be calculated for each cell first, and then replaced by the maximum one in the batch write
    double writeVoltageSquareSum;   // Sum of V^2 of non-identical pulses (for weight update energy calculation in subcircuits)

    double capacitance;                // the capacitance at the storage node. Use it to determine the voltage
    double chargeStorage=0;         // the charge stored at the capatitance;
    double chargeStoragePrev=0;
    double maxCharge = writeCurrentLTP*writePulseWidthLTP*maxNumLevelLTP;
    double voltageStorage = chargeStorage/capacitance; // the voltage at the storage node
	double conductance;	            // Current channel conductance (S) of the Transistor
	double conductancePrev;	    // Previous channel conductance (S) of the Transistor
	double conductanceRef;         // the conductance from the reference cell, which is (Gmax+Gmin)/2
    double currentRef;
    double maxConductance;	    // Maximum cell conductance (S)
	double minConductance;	    // Minimum cell conductance (S)
    int maxNumLevelLTP;	            // Maximum number of conductance states during LTP or weight increase
	int maxNumLevelLTD;	  
          // Maximum number of conductance states during LTD or weight decrease
	double xPulse;
    int numPulse;   // Number of write pulses used in the most recent write operation (Positive number: LTP, Negative number: LTD) (dynamic variable)
	double NL_LTP;		// LTP nonlinearity
	double NL_LTD;		// LTD nonlinearity
	double paramALTP;	// Parameter A for LTP nonlinearity
	double paramBLTP;	// Parameter B for LTP nonlinearity
	double paramALTD;	// Parameter A for LTD nonlinearity
	double paramBLTD;	// Parameter B for LTD nonlinearity
	double sigmaDtoD;	// Sigma of device-to-device variation on weight update nonliearity baseline
	double sigmaCtoC;	// Sigma of cycle-to-cycle variation on weight update
	
    bool nonlinearWrite;	// Consider weight update nonlinearity or not
    
	bool cmosAccess;	// Always true for the 2T1C cell
    double resistanceAccess;	// The resistance of two access transistors
    double widthAccessNMOS;
    double widthAccessPMOS;
    double widthAccessTransistor; // for the PCM
 

    /* device non-ideal effect */
    bool readNoise;	// Consider read noise or not
    double sigmaReadNoise;	// Sigma of read noise in gaussian distribution
	std::normal_distribution<double> *gaussian_dist;	// Normal distribution object
	std::normal_distribution<double> *gaussian_dist2;	// Normal distribution object
	std::normal_distribution<double> *gaussian_dist3;	// Normal distribution object
	std::normal_distribution<double> *gaussian_dist4;	// Normal distribution object
	std::normal_distribution<double> *gaussian_dist5;	// Normal distribution object
	std::normal_distribution<double> *gaussian_dist_maxConductance;	// Normal distribution object
	std::normal_distribution<double> *gaussian_dist_minConductance;	// Normal distribution object
	bool conductanceRangeVar;	// Consider variation of conductance range or not
	double maxConductanceVar;	// Sigma of maxConductance variation (S)
	double minConductanceVar;	// Sigma of minConductance variation (S)
    
    _3T1C(int x, int y);
	double Read(double voltage) ;
	void Write(double deltaWeightNormalized, double weight, double minWeight, double maxWeight);
	double GetMaxReadCurrent(void);
	double GetMinReadCurrent(void);

	void WriteEnergyCalculation(double wireCapCol);    
};


class HybridCell : public Cell {
public:
  // for the 3T1C+2PCM cell 
  _3T1C LSBcell;
  RealDevice MSBcell_LTP;
  RealDevice MSBcell_LTD;
  
  //for the 2T1FeFET cell
  //_3T1C LSBcell; // the LSBcell is the channel conductance of the FeFET
  //FeFET_MSB MSBcell_LTP; // the MSBcell is reflect the modulation of the 
  //FeFET_MSB MSBcell_LTD;  
    /* The MSB cell and LSB cell should use the same current */
  int significance; // the significance of the MSB cell
  double conductance;	                // total conductance of the hybrid cell
  double conductancePrev=0;	    // total conductance of the hybrid cell
  double readEnergy=0;	            // Dynamic variable for calculation of read energy (J)
  double writeEnergy=0;	            // Dynamic variable for calculation of write energy (J)
  double transferReadEnergy=0;
  double transferWriteEnergy=0;
  double transferEnergy=0;
  bool Analog;  // To Do analog read out or digital read
                // the analog read out is for FeFET hybrid precision cell
  bool Digital; 
  HybridCell(int x, int y);
  double ReadCell(void) ;
  double ReadMSB(void);
  void Write(double deltaWeightNormalized, double weight, double minWeight, double maxWeight) ;
  void WriteEnergyCalculation(double wireCapCol); 
  void WeightTransfer(double weightMSB_LTP, double weightMSB_LTD, double minWeight, double maxWeight, double wireCapCol);
};

class _2T1F : public AnalogNVM{
  public:
    double maxConductanceLSB;  // the conductance for the LSB part of the weight
    double minConductanceLSB;
    double maxConductanceMSB; // the conductance for the MSB part of the weight; need to calculate based on the cell conductance
    double minConductanceMSB;
    double conductanceLSB;   // only record the volatile and nonvolatile conductance. No other use
    double conductanceMSB; // for read/write, still use "conductance" 
    
    int maxNumLevelLTP_LSB; // the number of bits in the LSB cell
    int maxNumLevelLTD_LSB;
    int maxNumLevelLTP_MSB; // the number of bits in the MSB cell
    int maxNumLevelLTD_MSB;
    int prevMSBLevel=0; // Did not program the MSB cell at the initial state;
    int nowMSBLevel=0;
    bool transLTP = false; // if the cell is programmed to a higher MSB level;
    bool transLTD = false; // if the cell is programmed to a lower MSB level;
    
    
    double capacitance;                // the capacitance at the storage node. Use it to determine the voltage
    double chargeStorage=0;         // the charge stored at the capatitance;
    double chargeStoragePrev=0;
    double maxCharge = writeCurrentLTP*writePulseWidthLTP*maxNumLevelLTP;
    double writeCurrentLTP;  // Write current (A) for LTP or weight increase
    double writeCurrentLTD; // Write current (A) for LTP or weight increase
    
    double transVoltage[4] = {2,2.67,3.33,4}; // program voltage for the LTP
    double eraseVoltage;
    double transPulseWidth;
    double transEnergy=0;
    double transLatency=0;
    double eraseEnergy[4] = {4.072e-12,5.528e-12,6.837e-12,8.146e-12}; // store the energy consumption to erase/programm the MSB cell for each state;
    double programEnergy[4] = {2.036e-12,3.69e-12,5.692e-12,8.146e-12}; 
    
    double gateCapFeFET;	// Gate Capacitance of FeFET (F) use this to modulate the LSB conductance
    double widthAccessNMOS; 
    double widthAccessPMOS; 
    double widthFeFET;
    
    double NL_LTP;		// LTP nonlinearity
	  double NL_LTD;		// LTD nonlinearity
  	double paramALTP;	// Parameter A for LTP nonlinearity
  	double paramBLTP;	// Parameter B for LTP nonlinearity
  	double paramALTD;	// Parameter A for LTD nonlinearity
  	double paramBLTD;	// Parameter B for LTD nonlinearity
  	double sigmaDtoD;	// Sigma of device-to-device variation on weight update nonliearity baseline
  	double sigmaCtoC;	// Sigma of cycle-to-cycle variation on weight update
    double xPulse;
    //double numPulse;
    bool nonlinearWrite; 
    
    double transWriteEnergy=0; // the energy consumption when transfering the weight to MSB cell
                                                // calculated during the weight transfer;
 
    _2T1F(int x, int y);
  	double GetMaxReadCurrent() {return readVoltage * maxConductance;}
  	double GetMinReadCurrent() {return readVoltage * minConductance;}
  	double Read(double voltage) ;
  	void Write(double deltaWeightNormalized, double weight, double minWeight, double maxWeight);
    void WriteEnergyCalculation(double wireCapCol);
   // void WeightTransfer(double newConductance, char* mode);
    void WeightTransfer(void);
    
};

#endif
