# AFP-CKSAAP
AFP-CKSAAP: Prediction of Antifreeze Proteins Using Composition of k-Spaced Amino Acid Pairs with Deep Neural Network

# Requirements
- Python >= 3.5.4
- Tensorflow = 1.13.1
- Keras = 2.2.4

# Description
The deep learning model is implemented using Python on Keras (Tensorflow). The model file is uploaded with file named "CKSAAP_AFP.py". The model is composed of a trained network. 

# Dataset
The dataset is obtained from Kandaswamy et. al containing 481 antifreeze proteins and 9493 non-antifreeze proteins.
A sample fasta file for both AFP and non-AFP is uploaded in the folder "Sample_Dataset_Fas.fas" in fasta format.
# Features
The features from the dataset are extracted using CKSAAP encoding technique. The encoding scheme was utilized from iFeature web server which can be downloaded from 
(https://github.com/Superzchen/iFeature)

The CKSAAP feature descriptors can be extracted using the command 


**path/iFeature-master>python iFeature.py --file xyz/test-protein.txt --type CKSAAP --out xyz/test-protein-features.txt**


The CKSAAP feature encoding calculates the frequency of amino acid pairs separated by any k residues. The default value of k is 5. To change the the value of k a file named "placeholder.py" has been uploaded. The value of k can be replaced by an integer. The features used in this paper were extracted by selecting the value of k=8.

A sample feature file has been uploaded named "Sample_Features.csv"