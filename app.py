import warnings

from documents import *

warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import streamlit.components.v1 as components

from streamlit_option_menu import option_menu
import base64

from contents import *

# Set the page config
st.set_page_config(
    page_title="Forecasting Engine",
    page_icon="images/logo-white-rgb.png",
    layout="wide",
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
        options=[
            "Home",
            "Forecasting Engine",
            "Make Prediction",
            "History Tasks",
            "Documentation",
        ],
        # icons=["lightbulb", "graph-up", "card-list"],
        icons=[" ", " ", " ", " ", " "],
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

if choose == "Home":
    st.markdown(
        """
        <div class="content-title">👋🏻 Welcome</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.subheader("System Overview")
    st.markdown(
        """
    The **Forecasting Engine** is a user-friendly application that enables users to build Machine Learning models and perform time series forecasting simply by drag and drop. Through a simple interface and actions, users can upload datasets, configure models, adjust parameters, and generate predictions. 
    """
    )
    st.subheader("Quick Start")

    with open("quick_start/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()

    with open("quick_start/styles.css", "r", encoding="utf-8") as css_file:
        css_content = f"<style>{css_file.read()}</style>"

    components.html(
        f"""
            {css_content}
            {html_content}
        """,
        height=1800,
    )

elif choose == "Documentation":
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

    if toggle_quickstart:
        col1, col2 = st.columns([4.5, 1.5])
        with col2:
            quick_start_floating()
    else:
        col1, col2 = st.columns([5.9, 0.1])

    with col1:
        if choose == "Forecasting Engine":
            st.markdown(
                """
                <div class="content-title">💡 Forecasting Engine</div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("---")
            ml_modeling()

        elif choose == "Make Prediction":
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
