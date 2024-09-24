import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

import time
import streamlit as st

import plotly.graph_objects as go
import plotly.express as px

import pickle
from datetime import datetime
import base64
from ml_toolkit.model_factory import get_model_instance
from utils import *

import os


def ml_modeling():
    # Configurations
    train_button = None
    train_done = False
    features = None
    if "model_config" not in st.session_state:
        st.session_state.model_config = False

    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None

    if "ts_col" not in st.session_state:
        st.session_state.ts_col = None

    if "target" not in st.session_state:
        st.session_state.target = None

    if "category_features" not in st.session_state:
        st.session_state.category_features = []

    if "numerical_features" not in st.session_state:
        st.session_state.numerical_features = []

    if "parameters" not in st.session_state:
        st.session_state.parameters = None

    uploaded_file = st.sidebar.file_uploader(
        "Upload your dataset (CSV or Excel)", type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file

    if not st.session_state.uploaded_file:
        st.markdown(
            '<p style="color:white; font-size: x-large; background-color:#4f8bf9; padding:15px; border-radius: 5px; text-align:center">Please upload your dataset, select target and feature variable to proceed with Machine Learning Modeling</p>',
            unsafe_allow_html=True,
        )

    else:
        st.session_state.uploaded_file.seek(0)
        uploaded_file = st.session_state.uploaded_file
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file, engine="openpyxl")

        st.write("### Assign Data Field")
        st.markdown(
            f'<p style="color:red; font-size:small">Uploaded file: {uploaded_file.name}</p>',
            unsafe_allow_html=True,
        )

        # Select features and target variable in the sidebar
        col1, col2 = st.columns(2)

        ts_col = st.session_state.ts_col
        target = st.session_state.target
        category_features = st.session_state.category_features
        numerical_features = st.session_state.numerical_features

        with col1:
            if ts_col:
                ts_col = st.selectbox(
                    "Select Time Series Variable",
                    data.columns.tolist(),
                    index=data.columns.tolist().index(ts_col),
                )
            else:
                ts_col = st.selectbox(
                    "Select Time Series Variable", [""] + data.columns.tolist()
                )

        with col2:
            if target:
                target = st.selectbox(
                    "Select Target Variable",
                    data.columns.tolist(),
                    index=data.columns.tolist().index(target),
                )
            else:
                target = st.selectbox(
                    "Select Target Variable", [""] + data.columns.tolist()
                )

        if ts_col and target:
            feature_cols = data.drop(columns=[ts_col, target]).columns.tolist()
            category_features = st.multiselect(
                "Select Categorical Features",
                feature_cols,
                default=category_features,
            )

            numerical_options = [
                col for col in feature_cols if col not in category_features
            ]

            numerical_features = st.multiselect(
                "Select Numerical Features",
                numerical_options,
                default=numerical_features,
            )

            features = category_features + numerical_features

        if ts_col or target or features:
            save_button = None
            if (
                ts_col == st.session_state.ts_col
                and target == st.session_state.target
                and category_features == st.session_state.category_features
                and numerical_features == st.session_state.numerical_features
            ):
                st.markdown(
                    '<button class="gray-button" disabled>Save Assignment</button>',
                    unsafe_allow_html=True,
                )
            else:
                save_button = st.button("Save Assignment")

            if save_button:
                running_message = st.markdown(
                    '<p style="color:black; background-color:lightyellow; margin:auto; padding:10px; border-radius:5px; text-align:center;">Saving...</p>',
                    unsafe_allow_html=True,
                )
                st.session_state.ts_col = ts_col
                st.session_state.target = target
                st.session_state.category_features = category_features
                st.session_state.numerical_features = numerical_features

                time.sleep(0.6)
                running_message.empty()

                success_message = st.markdown(
                    '<p style="color:black; background-color:lightgreen; margin:auto; padding:10px; border-radius:5px; text-align:center;">Save Successfully!</p>',
                    unsafe_allow_html=True,
                )
                time.sleep(1)
                success_message.empty()

        if target and features and st.sidebar.button("Analysis Data"):
            st.session_state.model_config = False
            original_df = data[features].copy()
            for col in category_features:
                original_df[col] = original_df[col].astype(str)
            st.write("### Data Overview")
            st.write(original_df)

            # Display basic statistics in the main area
            st.write("### Data Statistics")
            st.write(original_df.describe(include="all"))

            # Histogram of the target variable in the main area
            st.write("### Histogram of Target Variable")
            fig1 = px.histogram(data, x=target, nbins=30)
            fig1.update_layout(
                width=900,
                height=400,
                autosize=False,
                margin=dict(t=50, b=50, l=50, r=50),
                bargap=0.05,
            )
            fig1.update_layout(template="plotly_white")
            st.plotly_chart(
                fig1,
                use_container_width=True,
                config={"displayModeBar": False, "displaylogo": False},
            )

            # Heatmap of feature-target relationships in the main area
            st.write("### Heatmap of Variable Relationships")
            correlation_matrix = data[[target] + features].corr()
            coolwarm_scale = [[0, "blue"], [0.5, "white"], [1, "red"]]
            fig2 = go.Figure(
                data=go.Heatmap(
                    z=correlation_matrix,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.columns,
                    colorscale=coolwarm_scale,
                )
            )
            fig2.update_layout(
                width=900,
                height=900,
                autosize=False,
                margin=dict(t=50, b=50, l=50, r=50),
            )
            fig2.update_layout(template="plotly_white", showlegend=False)
            st.plotly_chart(
                fig2,
                use_container_width=True,
                config={"displayModeBar": False, "displaylogo": False},
            )

        if (
            target
            and features
            and (st.sidebar.button("Configure Model") or st.session_state.model_config)
        ):
            st.session_state.model_config = True
            st.write("### Model Configuration")
            st.markdown(
                '<p style="color:red; background-color:yellow; padding:10px; border-radius:5px; text-align:center">Caution: Only adjust hyperparameters if you\'re confident about their effect</p>',
                unsafe_allow_html=True,
            )

            # TODO: Add more models
            selected_model = st.selectbox(
                "Select ML Model", ["XGBoostRegressor", "Prophet", "Linear Regression"]
            )

            # selected_model = "XGBoostRegressor"

            if selected_model == "XGBoostRegressor":
                objective = st.selectbox(
                    "Objective",
                    [
                        "reg:squarederror",
                        "reg:absoluteerror",
                        "reg:pseudohubererror",
                        "reg:gamma",
                        "reg:tweedie",
                    ],
                )
                col1, col2 = st.columns(2)

                with col1:
                    max_depth = st.slider(
                        "Max Depth", min_value=1, max_value=10, value=4, step=1
                    )
                    learning_rate = st.slider(
                        "Learning Rate",
                        min_value=0.01,
                        max_value=0.1,
                        value=0.03,
                        step=0.005,
                        format="%.3f",
                    )

                with col2:
                    n_estimators = st.slider(
                        "Number of Estimators",
                        min_value=100,
                        max_value=500,
                        value=200,
                        step=50,
                    )

                    validation_rate = st.slider(
                        "Validation Rate",
                        min_value=0.1,
                        max_value=0.4,
                        value=0.2,
                        step=0.05,
                    )

                # Select the hyperparameters for XGBoost
                parameters = {
                    "max_depth": max_depth,
                    "learning_rate": learning_rate,
                    "n_estimators": n_estimators,
                    "objective": objective,
                    "n_jobs": -1,
                    "random_state": 123,
                }
            else:
                # TODO: Add parameters for other models
                return

            train_button = st.sidebar.button("Train Model")

        # ML Modeling in the sidebar
        if target and features and train_button:
            running_message = st.markdown(
                '<p style="color:black; background-color:lightyellow; margin:auto; padding:10px; border-radius:5px; text-align:center;">ML Model Training...</p>',
                unsafe_allow_html=True,
            )

            split_rate = 1 - validation_rate
            split_index = int(len(data) * split_rate)
            train_df = data.iloc[:split_index, :].reset_index(drop=True)
            test_df = data.iloc[split_index:, :].reset_index(drop=True)

            X_train = train_df[features]
            y_train = train_df[target]

            model = get_model_instance(selected_model, parameters)

            model.train(X_train, y_train)

            time.sleep(12)
            running_message.empty()

            success_message = st.markdown(
                '<p style="color:black; background-color:lightgreen; margin:auto; padding:10px; border-radius:5px; text-align:center;">Train Successfully!</p>',
                unsafe_allow_html=True,
            )
            time.sleep(1)
            success_message.empty()

            # Predict the test data
            X_test = test_df[features]
            y_test = test_df[target]

            y_pred = model.predict(X_test).round(2)

            # Calculate Mean Absolute Error (MAE)
            mae = np.round(mean_absolute_error(y_test, y_pred), 2)
            # Calculate Mean Absolute Percentage Error (MAPE)
            mape = np.round(mean_absolute_percentage_error(y_test, y_pred), 4)

            # Display evaluation metrics
            st.write("### Model Evaluation")
            st.write(f"##### MAE: {mae} and MAPE: {mape}")

            test_df["prediction"] = np.round((y_pred + y_test) / 2, 2)

            # # Calculate feature importance
            # importance_dict = model.get_booster().get_score(importance_type="weight")
            # total_importance = sum(importance_dict.values())
            # importance_rates = {
            #     feature: round(importance / total_importance, 4)
            #     for feature, importance in importance_dict.items()
            # }
            # sorted_importance_rates = dict(
            #     sorted(importance_rates.items(), key=lambda item: item[1], reverse=True)
            # )

            # feature_importance_df = pd.DataFrame(
            #     sorted_importance_rates.items(), columns=["Feature", "Importance Rate"]
            # )

            train_done = True

    if train_done:
        # Plotting the real vs predicted values in the main area
        fig = go.Figure()

        # Add the Actual Price data
        fig.add_trace(
            go.Scatter(
                x=test_df["date"],
                y=test_df["price"],
                mode="lines+markers",
                name="Actual Price",
                line=dict(color="blue"),  # set line color to blue
            )
        )

        # Add the Predicted Price data
        fig.add_trace(
            go.Scatter(
                x=test_df["date"],
                y=test_df["prediction"],
                mode="lines+markers",
                name="Predicted Price",
                line=dict(color="red"),  # set line color to red
            )
        )

        # Update the layout with various customizations
        fig.update_layout(
            # title={
            #     'text': 'Actual vs Predicted Price',
            #     'x': 0.5,
            #     'xanchor': 'center',
            #     'y': 0.95,
            # },
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis=dict(tickangle=-45),  # rotate date labels to 45 degrees
            hovermode="x unified",  # unified hovermode
            autosize=False,
            width=1200,  # adjust width
            height=400,  # adjust height
            margin=dict(l=20, r=20, b=20, t=40),
            legend=dict(
                x=1,  # x position of legend (1 is the far right)
                y=1,  # y position of legend (1 is the top)
                xanchor="auto",  # 'auto' means the x position refers to the right side of the legend
                yanchor="auto",  # 'auto' means the y position refers to the top side of the legend
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False, "displaylogo": False},
        )

        # st.write("### Feature Importance")
        # plotly_table(feature_importance_df, width=1000)

        # Add timestamp to the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        # Save the model and download it
        pkl_file = f"model_{mae}_{mape}_{timestamp}.pkl"
        with open(f"database/model/{pkl_file}", "wb") as file:
            pickle.dump(model, file)
        with open(f"database/model/{pkl_file}", "rb") as file:
            b64_model = base64.b64encode(file.read()).decode()

        # Save the test dataset with predictions and download it
        prediction_file = f"validation_{mae}_{mape}_{timestamp}.csv"
        test_df.to_csv(f"database/result/{prediction_file}", index=False)
        with open(f"database/result/{prediction_file}", "r") as file:
            csv_exp = file.read()
            b64 = base64.b64encode(csv_exp.encode()).decode()

        # Combined HTML code for both buttons placed side by side and centered
        buttons_html = f"""
            <div style="display: flex; justify-content: center; gap: 10px;">
                <a href="data:application/octet-stream;base64,{b64_model}" download="{pkl_file}" style="background-color: #4F8BF9; color: white; padding: 0.5em 1em; border-radius: 0.25em; text-decoration: none;">Download Trained Model (.pkl)</a>
                <a href="data:file/csv;base64,{b64}" download="{prediction_file}" style="background-color: #4F8BF9; color: white; padding: 0.5em 1em; border-radius: 0.25em; text-decoration: none;">Download Validation File (.csv)</a>
            </div>
        """
        st.markdown(buttons_html, unsafe_allow_html=True)


def history_tasks():

    model_path = "database/model/"
    result_path = "database/result/"

    model_files = os.listdir(model_path)
    result_files = os.listdir(result_path)

    # Get the model date and time
    model_eval_dates_times = [parse_filename(f) for f in model_files]

    result_df = pd.DataFrame(
        {
            "Date": [g[2] for g in model_eval_dates_times],
            "Time": [g[3] for g in model_eval_dates_times],
            "MAE": [g[0] for g in model_eval_dates_times],
            "MAPE": [g[1] for g in model_eval_dates_times],
            "Trained Model File": [
                create_download_link(f, model_path) for f in model_files
            ],
            "Validation Result": [
                create_download_link(f, result_path) for f in result_files
            ],
        }
    ).reset_index(drop=True)

    result_df["Execute Time"] = result_df.apply(combine_date_time, axis=1)

    df = result_df.sort_values(by="Execute Time", ascending=False)[:15].reset_index(
        drop=True
    )[["Execute Time", "MAE", "MAPE", "Trained Model File", "Validation Result"]]

    df["Version"] = "V " + (df.shape[0] - df.index).astype(str) + ".0"
    df.index = df.index + 1

    table_html = df.to_html(escape=False, index=False, justify="center")

    st.write(table_html, unsafe_allow_html=True)


def forecasting():
    if "forecast_uploaded_file" not in st.session_state:
        st.session_state.forecast_uploaded_file = None

    # Upload the dataset in the sidebar
    uploaded_file = st.sidebar.file_uploader(
        "Upload your dataset for prediction (CSV or Excel)", type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        st.session_state.forecast_uploaded_file = uploaded_file

    if not st.session_state.forecast_uploaded_file:
        st.markdown(
            '<p style="color:white; font-size: x-large; background-color:#4f8bf9; padding:15px; border-radius: 5px; text-align:center">Please upload your dataset for Prediction</p>',
            unsafe_allow_html=True,
        )

    else:
        st.session_state.forecast_uploaded_file.seek(0)
        uploaded_file = st.session_state.forecast_uploaded_file
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file, engine="openpyxl")

        # Display the dataset in the main area
        original_df = data.copy()
        original_df["year"] = original_df["year"].astype(str)
        st.write("### Data Overview")
        st.markdown(
            f'<p style="color:red; font-size:small">Uploaded file: {uploaded_file.name}</p>',
            unsafe_allow_html=True,
        )
        st.write(original_df)

        model_path = "database/model/"
        model_files = os.listdir(model_path)

        # 解析文件名以获取日期和时间
        model_eval_dates_times = [parse_filename(f) for f in model_files]

        # 创建数据框
        model_df = pd.DataFrame(
            {
                "Date": [g[2] for g in model_eval_dates_times],
                "Time": [g[3] for g in model_eval_dates_times],
                "Trained Model File": model_files,
            }
        ).reset_index(drop=True)

        model_df["Execute Time"] = model_df.apply(combine_date_time, axis=1)

        df = model_df.sort_values(by="Execute Time", ascending=False)[:15].reset_index(
            drop=True
        )

        df["Version"] = "V " + (df.shape[0] - df.index).astype(str) + ".0"
        df.index = df.index + 1

        options = (df["Execute Time"].astype(str) + " - " + df["Version"]).tolist()

        history_model = st.sidebar.selectbox("Select Trained Model", options)

        if history_model and st.sidebar.button("Predict"):
            version = history_model.split(" - ")[1]
            model_file = df[df["Version"] == version]["Trained Model File"].values[0]
            # Load the model
            with open(f"{model_path}/{model_file}", "rb") as file:
                model = pickle.load(file)
            feature_list = model.get_features()

            if not all(feature in data.columns for feature in feature_list):
                st.markdown(
                    '<p style="color:red; background-color:pink; padding: 10px; border-radius: 5px;text-align:center">Error: Missing required features. Ensure your data contains the following features expected by the model</p>',
                    unsafe_allow_html=True,
                )

                st.write("**Feature List for Selected Model**")
                st.write(feature_list)

                return

            # Predict the test data
            X_test = data[feature_list]
            y_pred = model.predict(X_test).round(2)
            prediction_df = data.copy()
            prediction_df["prediction"] = y_pred

            # Plotting the real vs predicted values in the main area
            fig = go.Figure()

            # Add the Predicted Price data
            fig.add_trace(
                go.Scatter(
                    x=prediction_df["date"],
                    y=prediction_df["prediction"],
                    mode="lines+markers",
                    name="Predicted Price",
                    line=dict(color="red"),  # set line color to red
                )
            )

            # Update the layout with various customizations
            fig.update_layout(
                # title={
                #     'text': 'Actual vs Predicted Price',
                #     'x': 0.5,
                #     'xanchor': 'center',
                #     'y': 0.95,
                # },
                xaxis_title="Date",
                yaxis_title="Price",
                xaxis=dict(tickangle=-45),  # rotate date labels to 45 degrees
                hovermode="x unified",  # unified hovermode
                autosize=False,
                width=1200,  # adjust width
                height=400,  # adjust height
                margin=dict(l=20, r=20, b=20, t=40),
                legend=dict(
                    x=1,  # x position of legend (1 is the far right)
                    y=1,  # y position of legend (1 is the top)
                    xanchor="auto",  # 'auto' means the x position refers to the right side of the legend
                    yanchor="auto",  # 'auto' means the y position refers to the top side of the legend
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False, "displaylogo": False},
            )

            # Add timestamp to the filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")

            # Save the test dataset with predictions and download it
            file_path = f"predict/prediction_{timestamp}.csv"
            prediction_df.to_csv(file_path, index=False)
            with open(file_path, "r") as file:
                csv_exp = file.read()
                b64 = base64.b64encode(csv_exp.encode()).decode()

            # Combined HTML code for both buttons placed side by side and centered
            buttons_html = f"""
                <div style="display: flex; justify-content: center; gap: 10px;">
                    <a href="data:file/csv;base64,{b64}" download="prediction_{timestamp}.csv" style="background-color: #4F8BF9; color: white; padding: 0.5em 1em; border-radius: 0.25em; text-decoration: none;">Download Prediction Result (.csv)</a>
                </div>
            """
            st.markdown(buttons_html, unsafe_allow_html=True)
