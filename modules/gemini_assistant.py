
import streamlit as st
import json


def get_gemini_client():

    try:
        from google import genai

        api_key = st.secrets.get("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing from Streamlit secrets.")

        client = genai.Client(api_key=api_key)

        return client

    except Exception as error:
        raise RuntimeError(
            f"Unable to initialize Gemini: {error}"
        )


def explain_quantitative_results(
    question,
    research_context=None
):
    """
    Generate an AI explanation of quantitative research results.
    """

    client = get_gemini_client()

    model_name = st.secrets.get(
        "GEMINI_MODEL",
        "gemini-flash-latest"
    )

    context = json.dumps(
        research_context or {},
        indent=2,
        default=str
    )

    prompt = f"""
    You are a quantitative equity research assistant.

    Explain the following research question using clear,
    technically accurate financial and statistical language.

    Research Question:
    {question}

    Available Research Context:
    {context}

    Instructions:
    1. Explain the underlying financial concept.
    2. Interpret the supplied quantitative results.
    3. Discuss relevant assumptions and limitations.
    4. Avoid inventing data or performance figures.
    5. Do not present the output as investment advice.
    """

    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )

    return response.text


def summarize_factor_results(factor_results):

    question = (
        "Interpret the following factor research results. "
        "Discuss factor effectiveness, statistical significance "
        "where available, and limitations."
    )

    return explain_quantitative_results(
        question,
        factor_results
    )


def explain_stock_ranking(stock_ranking):

    question = (
        "Explain how the stocks are ranked using the supplied "
        "factor scores. Discuss the strengths and limitations "
        "of the composite ranking methodology."
    )

    return explain_quantitative_results(
        question,
        stock_ranking
    )
