# Applied-Math-Kalman-Filter-Thesis

This repository contains my work exploring **Kalman Filters** as part of my **Applied Mathematics senior thesis** at Colgate University.  
The goal of my thesis was to investigate state estimation techniques, analyzing how Kalman Filters can accurately estimate position and velocity from noisy sensor measurements.  

## Repository Contents

- **1D, 2D, and 3D Kalman Filter implementations**  
  - Includes single-sensor and multi-sensor configurations  
  - Demonstrates the effect of varying process and measurement noise (Q/R analysis)  
  - Scripts for analyzing RMSE performance across multiple runs  

- **2D Extended Kalman Filter (EKF)**  
  - Handles nonlinear dynamics such as air resistance in projectile motion  
  - Comparison against linear KF performance  

- **2D Unscented Kalman Filter (UKF)**  
  - Uses sigma points to approximate nonlinear transformations  
  - Tested against EKF and KF using Monte Carlo simulations  

All scripts are standalone and can be run independently to reproduce the results presented in my thesis experiments.
