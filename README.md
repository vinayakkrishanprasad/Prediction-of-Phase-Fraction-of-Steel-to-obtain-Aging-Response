# Prediction of Phase Fraction of Steel to Obtain Aging Response

This repository contains machine learning and deep learning models for predicting the phase fraction of steel based on its attributes, with the goal of understanding and optimizing the aging response of steel alloys. The project was undertaken at RV College of Engineering (June 2022 - December 2022) in collaboration with experts from the aerospace field.

## Project Overview

The aging response of steel is critical for various engineering applications, especially in aerospace. Predicting the phase fraction helps in tailoring the properties of steel for desired performance. This project leverages several regression techniques—from classical machine learning models to modern deep learning approaches—to model the relationship between steel attributes and its phase fraction.

## Key Features

- **Multiple Regression Models**: Implements Support Vector Machines (SVM), Decision Tree Regressor, Random Forest Regressor, and deep learning models (MLP Regressor using TensorFlow/Keras).
- **High Accuracy**: Achieved R² scores ranging from 0.82 to 0.93 for phase fraction prediction.
- **Data-Driven Approach**: Uses real experimental data (`data.xlsx`) consisting of 10 input attributes for each steel sample.

## Repository Structure

- `data.xlsx`: Dataset containing steel samples and their measured attributes.
- `Aging_new.ipynb`, `MLP_regressor.ipynb`, `Major_project.ipynb`: Jupyter notebooks with data exploration, preprocessing, model training, and evaluation.
- `.ipynb_checkpoints/`: Notebook checkpoints for autosave/recovery.
- Other notebook files: Experiments with various neural network architectures and model evaluation.

## How to Use

1. **Install Requirements**  
   Make sure you have Python 3.7+ and the following packages:
   - pandas
   - numpy
   - matplotlib
   - seaborn
   - scikit-learn
   - tensorflow
   - keras

   You can install them via:
   ```sh
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras
   ```

2. **Explore the Data**  
   Open any of the Jupyter notebooks in your favorite environment (VSCode, JupyterLab, Colab, etc.) to view the analysis and model code.

3. **Run the Models**  
   - Each notebook contains code cells for data preprocessing, model training, and evaluation.  
   - Results (including R² scores) are printed and visualized within the notebooks.

## Results

- **Best Model Performance**:  
  - Achieved R² scores in the range of **0.82–0.93** for phase fraction prediction using Random Forest and MLP regressors.
- **Feature Set**:  
  - 10 input attributes (X1 to X10) per sample, with X11 as the target phase fraction.

## Example Workflow

1. Load the dataset:
   ```python
   import pandas as pd
   df = pd.read_excel("data.xlsx")
   ```
2. Prepare features and labels:
   ```python
   X = df[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10']]
   Y = df['X11']
   ```
3. Train/test split and model training as shown in the notebooks.

## Acknowledgements

- **Institution**: RV College of Engineering
- **Duration**: June 2022 – December 2022
- **Collaboration**: Aerospace field experts

## License

This project is for academic and research purposes. Please contact the repository owner for other uses.

---

**Contact:**  
For questions or collaboration, please open an issue or contact [vinayakkrishanprasad](https://github.com/vinayakkrishanprasad).
