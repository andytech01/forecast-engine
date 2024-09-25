import streamlit as st


def system_overview():
    st.subheader("1.1. System Overview")
    st.markdown(
        """
    The **Forecasting Engine** is a user-friendly application that enables users to build machine learning models and perform time series forecasting. Through a simple interface, users can upload datasets, configure models, adjust parameters, and generate predictions. 

    Key features include visualizing model evaluation metrics, downloading trained models, and retrieving historical model performance.
    """
    )

    st.subheader("1.2. Key Features")
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
