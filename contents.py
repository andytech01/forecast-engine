from turtle import up
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

import streamlit as st

import plotly.graph_objects as go
import plotly.express as px

import pickle
from datetime import datetime
import base64

import os
import re


def plotly_table(data, width=500, use_container_width=False):
    ### Create a Plotly table
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["<b>" + col + "</b>" for col in data.columns],
                    fill_color="#4f8bf9",
                    font=dict(size=20, color="white"),  # Adjusting font size
                    align="center",
                    height=40,
                ),
                cells=dict(
                    values=[data[col] for col in data.columns],
                    fill_color="white",
                    font=dict(size=18, color="black"),  # Adjusting font size
                    align="center",
                    height=30,
                    line=dict(color="gray", width=1),
                ),
            )
        ]
    )

    # Adjusting the layout
    fig.update_layout(
        width=width,  # Adjust width
        margin=dict(t=10, b=10, l=10, r=10),  # Adjust margins
    )
    fig.update_layout(template="plotly_white")

    # Hide the default Plotly modebar
    config = {"displayModeBar": False, "displaylogo": False}
    st.plotly_chart(fig, use_container_width=use_container_width, config=config)


def ml_modeling():
    train_button = None
    train_done = False
    if "model_config" not in st.session_state:
        st.session_state.model_config = False

    # Upload the dataset in the sidebar
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None

    uploaded_file = st.sidebar.file_uploader(
        "Upload your dataset (CSV or Excel)", type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file

    if not st.session_state.uploaded_file:
        st.warning(
            "Please upload your dataset, select target and feature variable to proceed with the Data Analysis and Machine Learning"
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

        with col1:
            ts_col = st.selectbox(
                "Select Time Series Variable", [""] + data.columns.tolist()
            )

        with col2:
            target = st.selectbox(
                "Select Target Variable", [""] + data.columns.tolist()
            )

        if target and ts_col:
            feature_cols = data.drop(columns=[ts_col, target]).columns.tolist()

            category_features = st.multiselect(
                "Select Categorical Features", feature_cols, default=[]
            )

            if category_features:
                numerical_options = [
                    col for col in feature_cols if col not in category_features
                ]
                numerical_features = st.multiselect(
                    "Select Numerical Features", numerical_options, default=[]
                )
            else:
                numerical_features = st.multiselect(
                    "Select Numerical Features", feature_cols, default=[], disabled=True
                )

            features = category_features + numerical_features

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
                '<p style="color:red; background-color:yellow; padding: 10px; border-radius: 5px;">Caution: Only adjust hyperparameters if you\'re confident about their effect</p>',
                unsafe_allow_html=True,
            )

            selected_model = st.selectbox(
                "Select ML Models", ["XGBoost", "Prophet", "Linear Regression"]
            )

            if selected_model == "XGBoost":
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
                    # 使用 slider 替代 number_input
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

                # Select the hyperparameters for XGBoost

            train_button = st.sidebar.button("Train Model")

        # ML Modeling in the sidebar
        if target and features and train_button:
            split_rate = 0.8
            split_index = int(len(data) * split_rate)
            train_df = data.iloc[:split_index, :].reset_index(drop=True)
            test_df = data.iloc[split_index:, :].reset_index(drop=True)

            X_train = train_df[features]
            y_train = train_df[target]

            if selected_model == "XGBoost":
                model = xgb.XGBRegressor(
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    objective=objective,
                    n_jobs=-1,
                    random_state=123,
                )

            model.fit(X_train, y_train)

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

            # Calculate feature importance
            importance_dict = model.get_booster().get_score(importance_type="weight")
            total_importance = sum(importance_dict.values())
            importance_rates = {
                feature: round(importance / total_importance, 4)
                for feature, importance in importance_dict.items()
            }
            sorted_importance_rates = dict(
                sorted(importance_rates.items(), key=lambda item: item[1], reverse=True)
            )

            feature_importance_df = pd.DataFrame(
                sorted_importance_rates.items(), columns=["Feature", "Importance Rate"]
            )

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

        st.write("### Feature Importance")
        plotly_table(feature_importance_df, width=1000)

        # Add timestamp to the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        # Save the model and download it
        pkl_file = f"model/model_{mae}_{mape}_{timestamp}.pkl"
        with open(pkl_file, "wb") as file:
            pickle.dump(model, file)
        with open(pkl_file, "rb") as file:
            b64_model = base64.b64encode(file.read()).decode()

        # Save the test dataset with predictions and download it
        prediction_file = f"result/validation_{mae}_{mape}_{timestamp}.csv"
        test_df.to_csv(prediction_file, index=False)
        with open(prediction_file, "r") as file:
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


def parse_filename(filename):
    pattern = r"_([\d\.]*)_([\d\.]*)_(\d{8})_(\d{4})"
    match = re.search(pattern, filename)
    if match:
        return match.groups()
    return None, None


def combine_date_time(row):
    datetime_str = f"{row['Date']} {row['Time']}"
    return pd.to_datetime(datetime_str, format="%Y%m%d %H%M")


def create_download_link(filename, folder):
    filepath = os.path.join(folder, filename)
    with open(filepath, "rb") as f:
        bytes = f.read()
    b64 = base64.b64encode(bytes).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download</a>'


def history_tasks():
    model_files = os.listdir("model/")
    result_files = os.listdir("result/")

    # 解析文件名以获取日期和时间
    model_eval_dates_times = [parse_filename(f) for f in model_files]

    # 创建数据框
    result_df = pd.DataFrame(
        {
            "Date": [g[2] for g in model_eval_dates_times],
            "Time": [g[3] for g in model_eval_dates_times],
            "MAE": [g[0] for g in model_eval_dates_times],
            "MAPE": [g[1] for g in model_eval_dates_times],
            "Trained Model File": [
                create_download_link(f, "model/") for f in model_files
            ],
            "Validation Result": [
                create_download_link(f, "result/") for f in result_files
            ],
        }
    ).reset_index(drop=True)

    result_df["Execute Time"] = result_df.apply(combine_date_time, axis=1)

    df = result_df.sort_values(by="Execute Time", ascending=False)[:15].reset_index(
        drop=True
    )[["Execute Time", "MAE", "MAPE", "Trained Model File", "Validation Result"]]

    df["Version"] = "V " + (df.shape[0] - df.index).astype(str) + ".0"
    df.index = df.index + 1

    table_html = df.to_html(escape=False)

    st.write(table_html, unsafe_allow_html=True)


def forecasting():
    # Upload the dataset in the sidebar
    uploaded_file = st.sidebar.file_uploader(
        "Upload your dataset for prediction (CSV or Excel)", type=["csv", "xlsx"]
    )

    if not uploaded_file:
        st.warning("Please upload your dataset for Prediction")

    else:
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file, engine="openpyxl")

        # Display the dataset in the main area
        original_df = data.copy()
        original_df["year"] = original_df["year"].astype(str)
        st.write("### Data Overview")
        st.write(original_df)

        model_files = os.listdir("model/")

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

        history_model = st.sidebar.selectbox("Select Model", options)

        if history_model and st.sidebar.button("Predict"):
            version = history_model.split(" - ")[1]
            model_file = df[df["Version"] == version]["Trained Model File"].values[0]
            # Load the model
            with open(f"model/{model_file}", "rb") as file:
                model = pickle.load(file)

            # Predict the test data
            X_test = data.drop(columns=["date"])
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
