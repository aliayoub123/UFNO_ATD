# UFNO_ATD
U-FNO - an enhanced Fourier neural operator-based deep-learning model for atmospheric transport and dispersion 

In this work, we utilize the U-net enhanced Fourier neural operator model architecture, [U-FNO] (https://www.sciencedirect.com/science/article/pii/S0309170822000562), to emulate the atmospheric transport and dispersion calculation sof WSPEEDI in the context of real time nuclear emergency applications.


# Data sets
## Raw input data (U, V, and t)
https://drive.google.com/file/d/1wjXeUp64Yk8PBU7wB4v9kObjhiW8bBBn/view?usp=sharing 
## Raw output data (dose rate maps)
https://drive.google.com/file/d/1ol0GZv-pod6oHXO7LLxxjRGnmPfTb4gS/view?usp=sharing

Train set (n = 524):
input: sg_train_a.pt, output: sg_train_u.pt
input: dP_train_a.pt, output: dP_train_u.pt
## Output data
Test set (n = 92):
input: sg_test_a.pt, output: sg_test_u.pt
input: dP_test_a.pt, output: dP_test_u.pt

# Pre-trained models
The pre-trained models is available at: https://drive.google.com/file/d/191N_DdNJ2OWIklsVCdZc8_EIjU1ODZ-R/view?usp=sharing
