# Alphabet Soup Charity - Deep Learning Model Report
# Author: Matthew Adent

## Overview

The purpose of this was to build a binary classification model using a deep neural network that predicts whether an organization funded by Alphabet Soup will be successful. The model was trained on data from over 34,000 organizations, and it aimed to identify the features most associated with funding success.

I used TensorFlow, Keras, and NumPy to build and optimized a deep learning model with multiple hidden layers. The goal was to achieve an accuracy above 75%, and multiple valiant efforts were made to hit this accuracy, but I did not prevail. :(

---

## Results

### Data Preprocessing
- **Target Variable:** `IS_SUCCESSFUL` (binary: 1 if funding was successful, 0 otherwise)
- **Feature Variables:** All columns except `EIN`, `NAME`, `SPECIAL_CONSIDERATIONS`, `USE_CASE`, `ASK_AMT`
- **Dropped Variables:**
  - `EIN` and `NAME` (identifiers)
  - `SPECIAL_CONSIDERATIONS` (low variance)
  - `USE_CASE` (for my final attempt at optimization)
  - `ASK_AMT` was binned into categorical buckets and then dropped

### Compiling, Training, and Evaluating the Model
- **Model Layers and Activations:**
  - Attempt 1: 64 → 32 → 16 units (all `relu`)
  - Attempt 2: 128 → 64 → 32 (used `tanh` + `relu`, BatchNormalization, Dropout)
  - Attempt 3: 128 → 64 → 32 → 16, Dropout, `relu` activations, RMSprop optimizer
- **Output Layer:** 1 neuron with `sigmoid` activation for binary classification  

Despite everything, final accuracy on test data remained below 75%. Models generally achieved test accuracy in the range of 72–74%.

## Summary and Recommendation

The deep learning model performed reasonably well but did not meet the target 75% accuracy threshold. I'll never forgive myself for this. Nevertheless, I learned a lot from trying out different optimization tactics.

For anyone else that wants to take a crack at this, I recommend ensemble models (Random Forest, XGBoost). These models could outperform neural networks on the tabular, structured kind of data like we had, especially when feature interactions are important.
