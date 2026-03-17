import requests
import streamlit as st


API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
TEST_MESSAGE = "Hello!"


st.set_page_config(page_title="My AI Chat", layout="wide")
st.title("My AI Chat")


def load_hf_token() -> str | None:
    try:
        token = st.secrets["HF_TOKEN"]
    except Exception:
        return None

    if not isinstance(token, str):
        return None

    token = token.strip()
    return token or None


def fetch_test_reply(token: str) -> tuple[str | None, str | None]:
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": TEST_MESSAGE}],
        "max_tokens": 512,
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
    except requests.Timeout:
        return None, "The request timed out. Please try again in a moment."
    except requests.ConnectionError:
        return None, "Network error: unable to reach the Hugging Face API."
    except requests.RequestException as exc:
        return None, f"Request error: {exc}"

    if response.status_code == 401:
        return None, "Invalid Hugging Face token. Update `HF_TOKEN` in `.streamlit/secrets.toml`."
    if response.status_code == 429:
        return None, "Rate limit reached. Please wait and try again."
    if response.status_code >= 400:
        detail = response.text.strip() or f"HTTP {response.status_code}"
        return None, f"API error: {detail}"

    try:
        data = response.json()
        message = data["choices"][0]["message"]["content"].strip()
    except (ValueError, KeyError, IndexError, TypeError):
        return None, "The API returned an unexpected response format."

    if not message:
        return None, "The API returned an empty response."

    return message, None


hf_token = load_hf_token()

if not hf_token:
    st.error(
        "Missing Hugging Face token. Add `HF_TOKEN` to `.streamlit/secrets.toml` "
        "to run the chat test."
    )
    st.code('HF_TOKEN = "your_token_here"', language="toml")
else:
    st.subheader("API Test")
    st.write(f'User: "{TEST_MESSAGE}"')

    with st.spinner("Contacting the model..."):
        reply, error_message = fetch_test_reply(hf_token)

    if error_message:
        st.error(error_message)
    else:
        st.write("Assistant:")
        st.write(reply)
