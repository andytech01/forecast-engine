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
        .stCheckbox {
            padding-left: 5px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    .floating-quickstart {
        position: fixed;
        top: 100px;
        right: 20px;
        width: 300px;
        padding: 15px;
        background-color: #f9f9f9;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        border-radius: 10px;
        z-index: 100;
        font-size: 14px;
        color: #333;
        max-height: 70vh;  /* Max height relative to viewport height */
        overflow-y: auto;  /* Enable vertical scroll if content exceeds max-height */
    }
    .floating-quickstart h3 {
        margin-top: 0;
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

# Initialize session state for the toggle button
if "show_quickstart" not in st.session_state:
    st.session_state.show_quickstart = False

with st.sidebar:
    choose = option_menu(
        menu_title="",
        options=["ML Modeling", "Forecasting", "History Tasks", "Documentation"],
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

if choose == "Documentation":
    selected_page = st.sidebar.radio(
        "Select a page",
        ["Overview", "Quick Start", "Data Requirements", "Models"],
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

    if selected_page == "Data Requirements":
        st.markdown(
            '<div class="content-title">📋 Requirements</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        requirements()

    if selected_page == "Models":
        st.markdown(
            '<div class="content-title">🤖 Models</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        model_explanation()

    if selected_page == "API Reference":
        st.markdown(
            '<div class="content-title">📖 API Reference</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

else:
    # Toggle button for Quick Start visibility
    toggle_quickstart = st.sidebar.checkbox(
        "Show Quick Start Guide", value=st.session_state.show_quickstart
    )
    # Update session state based on toggle
    st.session_state.show_quickstart = toggle_quickstart

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
    if st.session_state.show_quickstart:
        quick_start_floating()
