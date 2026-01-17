# Neural Network-Based Prediction of Potential Energy Surfaces

## Problem Statement

Use a machine-learning algorithm to predict the potential energy surface and subsequently, reaction dynamics for the dissociation of formaldehyde while minimising computation time and reducing dependence on true quantum-mechanical methods of calculation


## Background Information

This project is an implementation of a neural network to generate a __machine-learned potential-energy surface__ for the photodissociation of formaldehyde. The photodissociation of formaldehyde can proceed through two mechanisms, proven experimentally, through a) radical or b) molecular pathways depending on the nature of the potential energy surface and the reaction dynamics involved. The nature of and local curvature of the potential energy surface determine the reaction dynamics and outcome. Traditional methods of simulation involving DFT/Coupled-Cluster methods are generally computationally and time intensive; which has led to an increase in machine-learning based solutions to speed up computational efficiency. The project was inspired by the work on photodissociation of nitrobenzene derivatives as stated in : 

__*https://www.researchgate.net/publication/385895885_Machine_Learned_Potential_Enables_Molecular_Dynamics_Simulation_to_Predict_the_Experimental_Branching_Ratios_in_the_NO_Release_Channel_of_Nitroaromatic_Compounds*__.


## Specifics

1) __Training Data__

The potential energy surface for formaldehyde dissociation is roughly estimated to be dependent on two parameters; the C-H bond length and H-H distance.

The energies of 196 variants of formaldehyde are generated using Psi4 in Python to compute DFT energies using M06-2x/6-31++G basis sets and functionals. The C-H bond length and H-H distance are varied in 14 equal steps from a to b and c to d respectively to generate the grid of training data.

2) __Neural Network__

A neural network was created using TensorFlow/Keras in Python to predict the potential energy surface for future reactions and was trained on the 196 point data grid. 

This problem presented a unique challenge as the goal of the model is to reduce overall DFT-dependence for PES generation; which demands a smaller training set. However, a smaller training set tends to lead to overfitting in complex models. The main issue here is that, in order to capture the intricacies and nuance in the PES, a relatively deep model is required. Thus, it ended up being a balancing game between model complexity and computational efficiency.

In the initial version, the model was trained using a k-fold validation split to reduce overfitting caused due to the small sample size(ModifiedNet.py). However, i ended up settling for a model with handpicked hyperparameters as the large sample space of possible hyperparameters and the small sample size led the model to pick very bad models that showed extremely tendencies to overfit training data instead of true extrapolative tendencies.


## Necessary Libraries

1) Psi4

2) NumPy

3) TensorFlow

4) SciPy

5) SciKit-Learn


## Basic Setup

__Terminal:__

activate __library-env__

cd projects

git clone

python ideal.py


## Problems Faced 

1) Lack of computational resources for calculating DFT training data, thus a smaller basis was chosen which leads to lower accuracy in the model.

2) The small sample size and model-complexity game showcased a unique set of problems while choosing model architecture.

3) Using C++ to increase efficiency and reduce time taken in the code.
   


## Improvements

1) The hyperparameters and model architecture must be appropriately tuned to ensure appropriate error reduction in the model.

2) The initial DFT data accuracy can be improved by using coupled-cluster methods and larger basis sets, at the expense of computational time.

3) Different model architectures can be explored to improve generation/ different ML algorithms such as GPRs, Gradient Boosting, Random Forests, e.t.c.

4) GPU acceleration should be employed to increase computational power.


## File Structure

1) cartesian.py is used to generate the different structures of formaldehyde for usage in the calculation

2) energytest.py generates the 14*14 grid of DFT energies for the different formaldehyde structures to train the ML model

3) ModifiedNet.py uses TensorFlow to train an ML model on the DFT grid by using keras-tuner to recursively improve on the model MSE and train the best possible model and                        visualise the PES in MatPlotLib (possible for a 2D PES) - Old Model

4) ideal.py uses TensorFlow to train an ML model on the DFT grid with a handpicked model architecture and hyperparameters

## Sample Graphs with Different Models

  <img width="453" height="426" alt="Screenshot 2025-10-29 at 2 25 57 PM" src="https://github.com/user-attachments/assets/a41974b5-bcad-41c5-9f53-bacbe04d0c74" />

  <img width="425" height="431" alt="Screenshot 2025-10-29 at 9 34 53 PM" src="https://github.com/user-attachments/assets/39914a6f-4800-42a9-9523-b0a500b4a3ba" />

<img width="453" height="360" alt="Screenshot 2026-01-17 at 4 12 12 PM" src="https://github.com/user-attachments/assets/87cfea64-7e99-42ba-ae5f-a9e3d2d883a5" />

<img width="426" height="392" alt="Screenshot 2026-01-17 at 4 11 06 PM" src="https://github.com/user-attachments/assets/186d2c26-269d-48fb-ae6b-a7a3c1665895" />

  

  
                


  
