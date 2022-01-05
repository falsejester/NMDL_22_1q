# MLP simlator (+NeuroSim) V3.0

The MLP+NeuroSim framework was developed by [Prof. Shimeng Yu's group](http://shimeng.ece.gatech.edu/) (Georgia Institute of Technology). The model is made publicly available on a non-commercial basis. Copyright of the model is maintained by the developers, and the model is distributed under the terms of the [Creative Commons Attribution-NonCommercial 4.0 International Public License](http://creativecommons.org/licenses/by-nc/4.0/legalcode)

This is the released version 3.0 (Mar. 1st, 2019) for the tool. This version extends the algoritihm weights from (0,1) in V2.0 to (-1,1) in V3.0. Besides, more optimization methods such as momentum method, Adagrad, RMSprop, Adam are added. The digital eNVMs (e.g. STT-MRAM) based synaptical array supports parallel read-out is introduced to reduce latency. Hybrid precision synapses with good linearity are added to provide better training accuracy

Developers: Pai-Yu Chen, Xiaochen Peng and Yandong Luo. 

If you have logistic questions or comments on the model, please contact Prof. Shimeng Yu (shimeng.yu@ece.gatech.edu), and if you have technical questions or comments, please contact Anni Lu(alu75@gatech.edu) or Yandong Luo (yandongluo@gatech.edu).  

This research is supported by NSF CAREER award, NSF/SRC E2CDA program, and ASCENT, one of the SRC/DARPA JUMP centers. 

If you use the tool or adapt the tool in your work or publication, you are required to cite the following reference:  

P.-Y. Chen, X. Peng, S. Yu, ※NeuroSim+: An integrated device-to-algorithm framework for benchmarking synaptic devices and array architectures,*§ IEEE International Electron Devices Meeting (IEDM)*, 2017, San Francisco, USA.

If you use the new features in this version (e.g. the advanced optimization methods such as momentum, Adagrad, RMSprop, Adam; the hybrid precision synapse such as 3T1C+NVM or 2T1F), you are required to cite the following reference:

Y. Luo, X. Peng and S. Yu, " MLP+NeuroSimV3.0: Improving On-chip Learning Performance with Device to Algorithm Optimizations," 2019 ACM the International Conference on Neuromorphic Systems (ICONS), ACM, New York, NY, USA. 

## File lists
1. MATLAB fitting script: `nonlinear_fit.m`
2. Nonlinearity-to-A table: `Documents/Nonlinearity-NormA.htm`
3. MNIST data: `MNIST_data.zip`
4. Manual: `Documents/Manual.pdf`
5. MLP Simulator (+NeuroSim): the rest of the files

## Installation steps (Linux)
1. Get the tool from GitHub
```
git clone https://github.com/neurosim/MLP_NeuroSim_V3.0.git
```

2. Extract `MNIST_data.zip` to it’s current directory
```
unzip MNIST_data.zip
```

3. Compile the codes
```
make
```

For the usage of this tool, please refer to the manual.

Updates on Jan. 20th, 2020: 
1. In sub-array, use linear-region transistor in MUX, Switch Matrix and across-transistor in array.
2. Calibrate FinFET technology library (<20nm)

Updates on Aug. 21st, 2021:
1. Add the hybrid precision synapse (3T1C+NVM, 2T1F) as device option. You can find more information on this in the ICONS'19 paper mentioned above.  
2. Add batch size as an input parameter (in param.cpp). 

## References related to this tool
1. Y. Luo, X. Peng and S. Yu, " MLP+NeuroSimV3.0: Improving On-chip Learning Performance with Device to Algorithm Optimizations," 2019 ACM the International Conference on Neuromorphic Systems (ICONS), ACM, New York, NY, USA.
2. P.-Y. Chen, S. Yu, "Technological benchmark of analog synaptic devices for neuro-inspired architectures," IEEE Design & Test, 2019.
3. P.-Y. Chen, X. Peng, S. Yu, “NeuroSim: A circuit-level macro model for benchmarking neuro-inspired architectures in online learning,” IEEE Trans. CAD,vol. 37, no. 12, pp. 3067-3080, 2018.
4. P.-Y. Chen, X. Peng, S. Yu, "NeuroSim+: An Integrated Device-to-Algorithm Framework for Benchmarking Synaptic Devices and Array Architectures," IEEE International Electron Devices Meeting (IEDM), 2017, San Francisco, USA.
5. P.-Y. Chen, X. Peng, S. Yu, "System-level benchmark of synaptic device characteristics for neuro-inspired computing," IEEE SOI-3D-Subthreshold Microelectronics Technology Unified Conference (S3S) 2017, San Francisco, USA.
6. P.-Y. Chen, S. Yu, "Partition SRAM and RRAM based synaptic arrays for neuro-inspired computing," IEEE International Symposium on Circuits and Systems (ISCAS)", 2016, Montreal, Canada.


