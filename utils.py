import pandas as pd
import base64
import plotly.graph_objects as go
import os
import streamlit as st
import re


def combine_date_time(row):
    datetime_str = f"{row['Date']} {row['Time']}"
    return pd.to_datetime(datetime_str, format="%Y%m%d %H%M")


def create_download_link(filename, folder):
    filepath = os.path.join(folder, filename)
    with open(filepath, "rb") as f:
        bytes = f.read()
    b64 = base64.b64encode(bytes).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download</a>'


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


def parse_filename(filename):
    pattern = r"_([\d\.]*)_([\d\.]*)_(\d{8})_(\d{4})"
    match = re.search(pattern, filename)
    if match:
        return match.groups()
    return None, None
