import pandas as pd
import streamlit as st
from transformers import pipeline
import torch
import asyncio

# Load the Excel file correctly and cache it
@st.cache_resource
def load_data():
    file_path = "C:/Users/91999/Downloads/PD dataset.xlsx"
   

    try:
        df = pd.read_excel(file_path, engine="openpyxl", usecols=[0, 1, 2, 3, 4, 5])  # Load only required columns
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

    # Clean column names
    df.columns = ["Date", "Thread Count", "Sender", "Recipient", "Subject", "Email Body"]
    
    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.dropna(subset=["Date"], inplace=True)

    return df

df = load_data()
if df is None:
    st.stop()  # Stop execution if data load fails

# Load summarization model and cache it
@st.cache_resource
def load_summarizer():
    device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
    return pipeline("summarization", model="facebook/bart-large-cnn", device=device, trust_remote_code=True)

summarizer = load_summarizer()

# Fetch Emails and Summarize Efficiently
async def fetch_emails(n=None, date_str=None):
    if date_str:
        search_date = pd.to_datetime(date_str, errors="coerce").date()
        emails_df = df[df["Date"].dt.date == search_date]
    else:
        emails_df = df.sort_values("Date", ascending=False).head(n or 5)

    if emails_df.empty:
        return []

    # Trim email body **before** summarization
    emails_df["Trimmed Body"] = emails_df["Email Body"].str[:1024]

    # Batch process summarization
    summaries = summarizer(
        emails_df["Trimmed Body"].tolist(),
        max_length=130,
        min_length=30,
        do_sample=False
    )

    # Construct output
    emails = [
        {
            "date": row["Date"].strftime("%Y-%m-%d"),
            "sender": row["Sender"],
            "recipient": row["Recipient"],
            "subject": row["Subject"],
            "summary": summaries[i]["summary_text"]
        }
        for i, (_, row) in enumerate(emails_df.iterrows())
    ]

    return emails

# Streamlit UI
st.title("📧 Gmail Genie - Email Summarizer")
user_input = st.text_input("Enter a date (YYYY-MM-DD) or  fetch number of recent emails:", key="command_input")

if st.button("Fetch Emails"):
    if user_input:
        try:
            pd.to_datetime(user_input, errors="raise")
            emails = asyncio.run(fetch_emails(date_str=user_input))
        except ValueError:
            n = int(''.join(filter(str.isdigit, user_input)) or 5)
            emails = asyncio.run(fetch_emails(n=n))
        
        if emails:
            for email in emails:
                st.subheader("📩 Email")
                st.write(f"**📅 Date:** {email['date']}")
                st.write(f"**✉️ Sender:** {email['sender']}")
                st.write(f"**📩 Recipient:** {email['recipient']}")
                st.write(f"**📜 Subject:** {email['subject']}")
                st.write(f"**📌 Summary:** {email['summary']}")
                st.markdown("---")
        else:
            st.warning("No emails found for the given input.")
    else:
        st.error("Please enter a valid date or a number.")
