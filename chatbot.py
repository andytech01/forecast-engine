import streamlit as st
from streamlit_chat import message
import pandas as pd
from llm_utils import chat_with_data_api


MAX_LENGTH_MODEL_DICT = {
    "gpt-4": 8191,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384,
}

st.markdown(
    """
    <style>
        footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_text():
    """Input text by the user"""
    input_text = st.text_input(label="Ask me your question.", value="", key="input")
    return input_text


def sidebar():
    """App sidebar content"""

    model = st.selectbox(
        label="Available Models",
        options=["gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-16k"],
        help="""The available models. Same prompt might return different results for
        different models. Epxerimentation is recommended.""",
    )

    temperature = st.slider(
        label="Temperature",
        value=0.0,
        min_value=0.0,
        max_value=2.0,
        step=0.01,
        help=(
            """Controls randomness. What sampling temperature to use, between 0 and 2.
            Higher values like 0.8 will make the output more random, while lower values
            like 0.2 will make it more focused and deterministic.
            It is recommended to alter this or `top_n` but not both"""
        ),
    )
    max_tokens = st.slider(
        label="Maximum length (tokens)",
        value=256,
        min_value=0,
        max_value=MAX_LENGTH_MODEL_DICT[model],
        step=1,
        help=(
            """The maximum number of tokens to generate in the chat completion.
            The total length of input tokens and generated tokens is limited by
            the model's context length."""
        ),
    )
    top_p = st.slider(
        label="Top P",
        value=0.5,
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        help=(
            """An alternative to sampling with temperature, called nucleus sampling,
            where the model considers the results of the tokens with top_p probability
            mass. So 0.1 means only the tokens comprising the top 10% probability
            mass are considered.
            It is recommended to alter this or `temperature` but not both"""
        ),
    )
    out_dict = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }
    return out_dict


def chat_with_data():

    st.title("Chat with, query and plot your own data")

    with st.sidebar:
        model_params = sidebar()
        memory_window = st.slider(
            label="Memory Window",
            value=3,
            min_value=1,
            max_value=10,
            step=1,
            help=(
                """The size of history chats that is kept for context. A value of, say,
                3, keeps the last three pairs of promtps and reponses, i.e. the last
                6 messages in the history."""
            ),
        )

    uploaded_file = st.file_uploader(label="Choose file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        prompt = f"""You are a python expert. You will be given questions for
            manipulating an input dataframe.
            The available columns are: `{df.columns}`.
            Use them for extracting the relevant data.
        """
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "system", "content": prompt}]
    else:
        df = pd.DataFrame([])

    # Storing the chat
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["Please upload your data"]

    if "past" not in st.session_state:
        st.session_state["past"] = []

    user_input = get_text()

    if (len(st.session_state["past"]) > 0) and (
        user_input == st.session_state["past"][-1]
    ):
        user_input = ""

    if ("messages" in st.session_state) and (
        len(st.session_state["messages"]) > 2 * memory_window
    ):
        # Keep only the system prompt and the last `memory_window` prompts/answers
        st.session_state["messages"] = (
            # the first one is always the system prompt
            [st.session_state["messages"][0]]
            + st.session_state["messages"][-(2 * memory_window - 2) :]
        )

    if user_input:
        if df.empty:
            st.warning("Dataframe is empty, upload a valid file", icon="⚠️")
        else:
            st.session_state["messages"].append({"role": "user", "content": user_input})
            response = chat_with_data_api(df, **model_params)
            st.session_state.past.append(user_input)
            if response is not None:
                st.session_state.generated.append(response)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": response}
                )

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            if i - 1 >= 0:
                message(
                    st.session_state["past"][i - 1], is_user=True, key=str(i) + "_user"
                )


if __name__ == "__main__":
    chat_with_data()
