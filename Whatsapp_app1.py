import streamlit as st
from transformers import pipeline
import re
from collections import Counter

pipe = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")

def extract_messages(file_content):
    messages = []
    for line in file_content.decode("utf-8").splitlines():
        match = re.match(r'^\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2} - (.*?): (.*)', line)
        if match:
            sender, message = match.groups()
            messages.append(message)
    return messages

def analyze_sentiments(messages):
    sentiments = []
    for message in messages:
        result = pipe(message)[0]
        sentiments.append((message, result['label'], result['score']))
    return sentiments

def display_statistics(sentiments):
    labels = [label for _, label, _ in sentiments]
    label_counts = Counter(labels)
    
    st.subheader("Sentiment Distribution")
    st.bar_chart(label_counts)
    
    st.write("### Overall Sentiment Counts")
    for label, count in label_counts.items():
        st.write(f"**{label.capitalize()}**: {count}")

st.set_page_config(page_title="WhatsApp Chat Sentiment Analysis", page_icon="📲", layout="centered")
st.title("WhatsApp Chat Sentiment Analysis")
st.write("Upload a WhatsApp chat text file to analyze the sentiment of each message.")

uploaded_file = st.file_uploader("Choose a WhatsApp chat file", type="txt")

if uploaded_file is not None:
    messages = extract_messages(uploaded_file.read())
    
    if messages:
        st.write("Analyzing sentiments... Please wait.")
        sentiments = analyze_sentiments(messages)
        
        display_statistics(sentiments)
        
        st.write("### Message-by-Message Analysis")
        for message, label, score in sentiments:
            sentiment_color = "green" if label == "joy" else "red" if label == "anger" else "blue"
            st.write(f"**Message:** {message}")
            st.markdown(f"<span style='color:{sentiment_color};font-weight:bold;'>{label.capitalize()}</span>, Confidence: {score:.2f}", unsafe_allow_html=True)
            st.write("---")
    else:
        st.write("No messages found in the file. Make sure it follows the standard WhatsApp format.")

st.write("Created by Nikhil Dwivedi")
st.write("Model Used: SamLowe/roberta-base-go_emotions")
