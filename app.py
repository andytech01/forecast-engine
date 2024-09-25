import warnings

from documents import *

warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
from streamlit_option_menu import option_menu
import base64

from contents import *

# Set the page config
st.set_page_config(
    page_title="Forecasting Engine",
    page_icon="images/logo-white-rgb.png",
    # layout="wide",
    initial_sidebar_state="expanded",
)

### Load the custom CSS file
with open("css/styles.css", "r") as f:
    styles_css = f.read()
    st.markdown(
        f"<style>{styles_css}</style>",
        unsafe_allow_html=True,
    )

### Load the sidebar CSS file
with open("css/sidebar.css", "r") as f:
    sidebar_css = f.read()
    st.sidebar.markdown(f"<style>{sidebar_css}</style>", unsafe_allow_html=True)

### Load the main area CSS file
with open("css/content.css", "r") as f:
    content_css = f.read()
    st.markdown(f"<style>{content_css}</style>", unsafe_allow_html=True)


st.markdown(
    """
    <style>
        .stRadio {
            padding-left: 40px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

### Sidebar content
with open("images/logo.png", "rb") as img_file:
    logo = base64.b64encode(img_file.read()).decode()

st.sidebar.markdown(
    f"""
    <div style="display:flex; flex-direction:column">
        <div style="text-align: center; margin: 2em 0;">
            <img src="data:image/png;base64,{logo}" width="200"/>
        </div>
        <!-- <div class='big-font'>Forecasting Engine</div> -->
    </div>""",
    unsafe_allow_html=True,
)

with st.sidebar:
    choose = option_menu(
        menu_title="",
        options=["ML Modeling", "Forecasting", "History Tasks", "Docuemntation"],
        # icons=['lightbulb', 'graph-up', 'card-list'],
        icons=[" ", " ", " ", " "],
        menu_icon="app-indicator",
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "#ebf6ff"},
            "icon": {"color": "blue", "font-size": "23px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#ededed",
            },
            "nav-link-selected": {"background-color": "#4f8bf9"},
        },
    )
st.sidebar.markdown("---")

if choose == "ML Modeling":
    st.markdown(
        """
        <div class="content-title">💡 Machine Learning Modeling</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    ml_modeling()

elif choose == "Forecasting":
    st.markdown(
        '<div class="content-title">📈 Forecasting</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    forecasting()

elif choose == "History Tasks":
    st.markdown(
        '<div class="content-title">📜 History Tasks</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    history_tasks()
elif choose == "Docuemntation":
    selected_page = st.sidebar.radio(
        "Select a page",
        ["Overview", "Quick Start", "Requirements", "Models", "API Reference"],
    )
    if selected_page == "Overview":
        st.markdown(
            '<div class="content-title">📄 Overview</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        system_overview()
    if selected_page == "Quick Start":
        st.markdown(
            '<div class="content-title">📘 Quick Start for Forecasting Engine</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        quick_start()

    if selected_page == "Requirements":
        st.markdown(
            '<div class="content-title">📋 Requirements</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

    if selected_page == "Models":
        st.markdown(
            '<div class="content-title">🤖 Models</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

    if selected_page == "API Reference":
        st.markdown(
            '<div class="content-title">📖 API Reference</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
