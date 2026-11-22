import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="Product Review Sentiment Analyzer", layout="wide")
st.title("Product Review Sentiment Analyzer")
st.write("Upload a CSV of product reviews to classify sentiments in batches of 20 and summarize common complaints.")

api_key = os.getenv("GEMINI_API_KEY") or st.text_input("Enter Gemini API Key", type="password")
uploaded_file = st.file_uploader("Upload Reviews CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview:", df.head())
    text_col = st.selectbox("Select review text column", df.columns)

    if st.button("Analyze Reviews"):
        if not api_key:
            st.error("Please provide a Gemini API key.")
            st.stop()

        client = genai.Client(api_key=api_key)
        reviews = df[text_col].dropna().astype(str).tolist()
        sentiments = []
        progress = st.progress(0)
        batch_size = 20

        for i in range(0, len(reviews), batch_size):
            batch = reviews[i:i + batch_size]
            prompt = (
                "Classify each product review below strictly as Positive, Negative, or Neutral.\n"
                f"Return ONLY a valid JSON list of strings in the exact same order (length {len(batch)}).\n\n"
                "Reviews:\n" + json.dumps(batch, ensure_ascii=False)
            )
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            batch_labels = json.loads(resp.text)
            if len(batch_labels) != len(batch):
                batch_labels = (batch_labels + ["Neutral"] * len(batch))[:len(batch)]
            sentiments.extend(batch_labels)
            progress.progress(min(1.0, (i + batch_size) / len(reviews)))

        df_result = pd.DataFrame({"review": reviews, "sentiment": sentiments})
        st.subheader("Classified Reviews")
        st.dataframe(df_result, use_container_width=True)

        counts = df_result["sentiment"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.subheader("Sentiment Distribution")
        st.pyplot(fig)

        neg_reviews = df_result[df_result["sentiment"].str.lower() == "negative"]["review"].tolist()
        st.subheader("Common Complaints Summary (Negative Reviews)")
        if neg_reviews:
            summary_prompt = (
                "Summarize the common customer complaints from these negative product reviews into concise bullet points:\n\n"
                + "\n".join(f"- {r}" for r in neg_reviews[:100])
            )
            summary_resp = client.models.generate_content(model="gemini-2.5-flash", contents=summary_prompt)
            st.write(summary_resp.text)
        else:
            st.info("No negative reviews found in the uploaded dataset.")
