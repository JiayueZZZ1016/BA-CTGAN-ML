# BA-CTGAN-ML
Recycling potential, environmental impacts and economic benefits of municipal solid waste incineration bottom ash in China


# 🤖 Machine Learning for Predicting Incinerator Bottom Ash (BA) Generation

[](https://www.python.org/downloads/release/python-390/)
[](https://jupyter.org/)
[](https://www.google.com/search?q=https://ml.azure.com/)

This repository contains the source code and data for the machine learning section of the paper: **"Mapping the Recycling Potential of Bottom Ashes from Waste-to-Energy Plants toward Circular Economy: Evidence from China"**. 

The primary challenge addressed by this model was the limited availability of data on Incinerator Bottom Ash (IBA) generation, with samples covering only 18% of China's Waste-to-Energy (WtE) facilities. To overcome this, we developed a machine learning framework to generate high-resolution, nationwide predictions from the limited data. 

## 📖 AI/ML Workflow

Our approach integrates data augmentation using a Generative Adversarial Network (GAN) with several machine learning models to accurately predict the IBA generation ratio at the facility level. 

****
<img width="1980" height="2687" alt="image" src="https://github.com/user-attachments/assets/06351d03-24b8-44c9-bff8-759cd0e3079d" />

*Figure: Overview of the Machine Learning Workflow *

### 1\. Data Augmentation

To enhance the model's generalization capabilities with a small initial sample size, we employed a Conditional Tabular Generative Adversarial Network (CTGAN) to create synthetic data samples. The modeling process was tested with datasets augmented by 50, 100, and 150 synthetic samples to find the optimal data volume.

### 2\. Predictive Modeling

We evaluated eight high-performing machine learning algorithms to predict the IBA generation ratio:

  * Extreme Gradient Boosting (XGBoost)
  * Category Gradient Boosting (CatBoost) 
  * Support Vector Machine (SVM) 
  * Light Gradient Boosting Machine (LightGBM) 
  * Gradient Boosting Machine (GBM) 
  * K-Nearest Neighbors (KNN)
  * Random Forest (RF)
  * Neural Networks (NN)

The input variables included 10 socioeconomic characteristics and 4 WtE technology parameters. Model performance was assessed using the coefficient of determination ($R^2$), root-mean-square error (RMSE), and mean absolute error (MAE). Hyperparameter optimization was performed using 5-fold cross-validation with Bayesian optimization. 

### 3\. Model Interpretability

To understand the factors influencing IBA generation, Shapley Additive Explanations (SHAP) values were calculated to measure the relative contribution and marginal effects of each input feature on the model's output. 

## 📊 Key Results

  * **Best Performing Model**: The **XGBoost** model consistently outperformed other algorithms, achieving the highest accuracy with a dataset augmented by 100 synthetic samples. 
  * **Performance Improvement**: Data augmentation significantly improved prediction accuracy, increasing the test set $R^2$ value to **0.87**, a 21% improvement over the non-augmented baseline ($R^2=0.72$).
  * **Influential Features**: The most important predictive features were socioeconomic and environmental factors, including total retail sales, electricity generation, and average annual temperature, which collectively accounted for over 70% of the model's predictive power. 

## 🗂️ Repository Contents

  * `main_code.ipynb`: A Jupyter Notebook containing the Python code for data augmentation, model training, evaluation, and feature importance analysis.
  * `data_lz.xlsx`: The original, raw dataset used for training the models.
  * `augmented_data.csv`: The dataset after applying CTGAN for data augmentation.
  * `README.md`: An overview of the project and methodology.

## ✍️ How to Cite

If you use this code or the findings from our work, please cite the original paper:

Zhang, J., Zhang, Y., Leong, Z. H., Zhang, Y., Chen, T., Fei, F., & Wen, Z. (2025). Mapping the Recycling Potential of Bottom Ashes from Waste-to-Energy Plants toward Circular Economy: Evidence from China. [cite\_start]*Environmental Science & Technology*. [cite: 8]
