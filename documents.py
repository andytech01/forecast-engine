import streamlit as st
import pandas as pd


def system_overview():
    st.subheader("1. System Overview")
    st.markdown(
        """
    The **Forecasting Engine** is a user-friendly application that enables users to build machine learning models and perform time series forecasting. Through a simple interface, users can upload datasets, configure models, adjust parameters, and generate predictions. 

    Key features include visualizing model evaluation metrics, downloading trained models, and retrieving historical model performance.
    """
    )

    st.subheader("2. Key Features")
    st.markdown(
        """
    - **Dataset Upload**: Users can upload `.csv` or `.xlsx` files, then select the time series column and target variable for modeling.
    - **Machine Learning Modeling (ML Modeling)**: The application supports multiple machine learning models (e.g., XGBoost, Prophet, Linear Regression). Users can configure model hyperparameters through an intuitive sidebar interface.
    - **Forecasting**: After training the model, users can apply it to new datasets to generate predictions, view actual vs predicted results, and download the prediction files.
    - **History Tasks**: Users can review past trained models and tasks, as well as download model files and validation results for future use.
    - **Documentation**: This section provides detailed system documentation and guides to help users understand the features, metrics, and models available in the forecasting engine.
    """
    )


def quick_start():
    st.write("**Step 1: Upload Data**")
    st.markdown(
        """
    - Navigate to the sidebar and upload your dataset in either `.csv` or `.xlsx` format.
    - The data preview will be displayed in the main area once uploaded.
    """
    )

    st.write("**Step 2: Assign Data Fields**")
    st.markdown(
        """
    - Select the time series variable and target variable from your dataset.
    - Optionally, assign categorical and numerical features to improve the model's performance.
    """
    )

    st.write("**Step 3: Analyze Data**")
    st.markdown(
        """
    - After assigning the features and target, click **Analyze Data** to inspect the dataset, view summary statistics, and visualize relationships between variables.
    """
    )

    st.write("**Step 4: Configure Models**")
    st.markdown(
        """
    - Choose a machine learning model (e.g., XGBoost) and adjust hyperparameters such as max depth, learning rate, and number of estimators.
    - After configuration, train the model by clicking the **Train Model** button.
    """
    )

    st.write("**Step 5: Evaluate the Model**")
    st.markdown(
        """
    - The application will display model evaluation metrics such as **Mean Absolute Error (MAE)** and **Accuracy**, along with feature importance charts and prediction results.
    """
    )

    st.write("**Step 6: Forecasting**")
    st.markdown(
        """
    - To make predictions on new data, upload the dataset under the **Forecasting** section and choose a previously trained model.
    - The predictions will be displayed alongside actual data, and users can download the results as a `.csv` file.
    """
    )

    st.write("**Step 7: Review History Tasks**")
    st.markdown(
        """
    - Visit the **History Tasks** section to review previous models and download files for future reference.
    """
    )


def quick_start_floating():
    st.markdown(
        """
        <div class="floating-quickstart">
        <h4 style="text-align:center;">Quick Start</h4>

        <details>
            <summary>Step 1: Upload Data</summary>
            <ul>
                <li>Navigate to the sidebar and upload your dataset in either <span style="font-weight:bold; color:#4f8bf9">.csv</span> or <span style="font-weight:bold; color:#4f8bf9">.xlsx</span> format.</li>
                <li>The data preview will be displayed in the main area once uploaded.</li>
            </ul>
        </details>

        <details>
            <summary>Step 2: Assign Data Field</summary>
            <ul>
                <li>Select the time series variable and target variable from your dataset.</li>
                <li>Optionally, assign categorical and numerical features to improve the model's performance.</li>
            </ul>
        </details>

        <details>
            <summary>Step 3: Analyze Data</summary>
            <ul>
                <li>After assigning the features and target, click <span style="font-weight:bold">Analyze Data</span> to inspect the dataset, view summary statistics, and visualize relationships between variables.</li>
            </ul>
        </details>

        <details>
            <summary>Step 4: Configure Model</summary>
            <ul>
                <li>Choose a machine learning model (e.g., XGBoost) and adjust hyperparameters such as max depth, learning rate, and number of estimators.</li>
                <li>After configuration, train the model by clicking the <span style="font-weight:bold">Train Model</span> button.</li>
            </ul>
        </details>

        <details>
            <summary>Step 5: Evaluate the Model</summary>
            <ul>
                <li>The application will display model evaluation metrics such as <span style="font-weight:bold">Mean Absolute Error (MAE)</span> and <span style="font-weight:bold">Accuracy</span>, along with feature importance charts and prediction results.</li>
            </ul>
        </details>

        <details>
            <summary>Step 6: Forecasting</summary>
            <ul>
                <li>To make predictions on new data, upload the dataset under the <span style="font-weight:bold">Forecasting</span> section and choose a previously trained model.</li>
                <li>The predictions will be displayed alongside actual data, and users can download the results as a <span style="font-weight:bold;color:#4f8bf9">.csv</span> file.</li>
            </ul>
        </details>

        <details>
            <summary>Step 7: Review History Tasks</summary>
            <ul>
                <li>Visit the <span style="font-weight:bold">History Tasks</span> section to review previous models and download files for future reference.</li>
            </ul>
        </details>
        </div>
    """,
        unsafe_allow_html=True,
    )


def requirements():
    st.subheader("1. Dataset Structure")
    st.markdown(
        """
    Before using the app, ensure your dataset is properly formatted. Below is an example of a time-series dataset suitable for forecasting. You can upload your dataset in either `.csv` or `.xlsx` format. 
    """
    )
    # Create the example dataset using pandas
    example_data = {
        "Date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
        "Sales": [150, 200, 180, 220, 170],
        "Category": [
            "Electronics",
            "Furniture",
            "Electronics",
            "Furniture",
            "Electronics",
        ],
        "Region": ["North", "South", "East", "West", "North"],
        "Discount": [5.0, 10.0, 7.5, 12.0, 6.0],
    }

    df = pd.DataFrame(example_data)

    # Display the dataset as a table
    st.table(df)

    st.markdown(
        """
    - **Date**: The time series variable representing the date of each record.
    - **Sales**: The target variable that you want to forecast (e.g., daily sales).
    - **Category**: A categorical feature representing the product category.
    - **Region**: A categorical feature representing the region of the sales.
    - **Discount**: A numerical feature representing the discount applied to the sales.

    Ensure your dataset **at least has the time series column and the target column**. Better to have **numerical features** (e.g., Sales, Discount) and **categorical features** (e.g., Category, Region) to improve model performance.
    """
    )

    st.subheader("2. Step-by-Step Instructions")

    st.write("**Step 1: Upload Data**")
    st.markdown(
        """
    In the sidebar, upload your dataset in `.csv` or `.xlsx` format. Once uploaded, a preview of the dataset will appear in the main area.
    """
    )

    st.write("**Step 2: Assign Data Fields**")
    st.markdown(
        """
    - **Select Time Series Variable**: Choose the column that represents the time (e.g., **Date**).
    - **Select Target Variable**: Choose the variable you want to forecast (e.g., **Sales**).
    - **Assign Categorical and Numerical Features**:
      - Choose categorical features (e.g., **Category**, **Region**).
      - Choose numerical features (e.g., **Discount**) to improve model performance.
    """
    )

    st.write("**Step 3: Analyze Data**")
    st.markdown(
        """
    - Click the **Analyze Data** button after assigning fields to view summary statistics, data distribution, and correlations.
    """
    )

    st.write("**Step 4: Configure Models**")
    st.markdown(
        """
    - Choose a machine learning model (e.g., **XGBoost**) and configure hyperparameters such as **Max Depth**, **Learning Rate**, and **Number of Estimators**.
    - Click **Train Model** to start the training process.
    """
    )

    st.write("**Step 5: Evaluate the Model**")
    st.markdown(
        """
    - After training, the app will display metrics such as **Mean Absolute Error (MAE)**, **Mean Absolute Percentage Error (MAPE)**, and **Feature Importance** to help evaluate model performance.
    """
    )

    st.write("**Step 6: Forecasting on New Data**")
    st.markdown(
        """
    - In the **Forecasting** section, upload a new dataset with the same structure as your training data, and apply a trained model to generate forecasts.
    - Predictions and visualizations will be displayed, and you can download the results in `.csv` format.
    """
    )

    st.write("**Step 7: Review History Tasks**")
    st.markdown(
        """
    - In the **History Tasks** section, review previously trained models, view evaluation metrics, and download models or prediction results for future use.
    """
    )


def model_explanation():
    # Table of Contents
    st.markdown("## Table of Contents")
    st.markdown(
        """
    1. [Machine Learning Models](#1-machine-learning-models)
    2. [Evaluation Metrics](#2-evaluation-metrics)
    3. [Feature Importance](#3-feature-importance)
    4. [Forecasting Methods](#4-forecasting-methods)
    """
    )
    st.markdown("---")
    st.subheader("1. Machine Learning Models")
    st.markdown("*Keep adding more models*")
    st.markdown(
        """
    The **Forecasting Engine** supports several machine learning models, each designed for different types of data and forecasting needs.

    - **XGBoostRegressor**: A powerful gradient boosting algorithm optimized for structured/tabular data. It supports various objectives like `reg:squarederror` (mean squared error) and `reg:absoluteerror` (mean absolute error), making it highly flexible for regression tasks.

        **Key Parameters**:
        - **max_depth**: Maximum depth of a tree.
        - **learning_rate**: Step size shrinkage used to prevent overfitting.
        - **n_estimators**: The number of trees to be built.

    - **Prophet**: Developed by Facebook, Prophet is used for time series forecasting, particularly suited for datasets with daily or seasonal patterns. It can handle missing data and outliers well.

    - **Linear Regression**: A basic model that assumes a linear relationship between the input features and the target variable. It’s suitable for datasets where the relationships between variables are more straightforward.
    """
    )

    st.subheader("2. Evaluation Metrics")
    st.markdown(
        "To ensure the accuracy and reliability of predictions, the application provides several evaluation metrics:"
    )

    st.markdown(
        "- **Mean Absolute Error (MAE)**: This metric measures the average magnitude of the errors in a set of predictions, without considering their direction. ***A lower MAE values indicate better model performance***"
    )

    st.latex(
        r"""
    MAE = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|
    """
    )

    st.markdown(
        "- **Accuracy (1 - MAPE)**: Instead of using MAPE directly, we calculate accuracy as \( 1 - MAPE \). This provides a percentage-based measure of how close the predicted values are to the actual values. ***A higher accuracy indicates better model performance***"
    )

    st.latex(
        r"""
    Accuracy = 1 - \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|
    """
    )

    st.subheader("3. Feature Importance")
    st.markdown(
        """
    Understanding the importance of features in a model is key to interpreting results. The app visualizes feature importance for models, allowing users to understand which variables have the most significant impact on the predictions.

    **Feature Importance**: Feature importance reflects how much impact a particular feature has on the model's predictions. Features with higher importance rate contribute more significantly to the model's decisions, meaning they play a larger role in determining the output. Understanding feature importance helps in interpreting the model and identifying key factors driving the predictions.
    """
    )

    st.subheader("4. Forecasting Methods")
    st.markdown(
        """
    Forecasting in the app leverages trained Machine Learning models to predict future values based on new data. The process includes the following steps:

    - **Trained Model Selection**: After uploading new data, users can select a previously trained model from the sidebar to apply to the dataset.
    - **Prediction Output**: The predictions will be displayed alongside with a graph.
    """
    )
