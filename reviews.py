import streamlit as st
import pandas as pd
from langchain_community.llms import Cohere
import cohere

# Set page configuration (must be the first Streamlit command)
st.set_page_config(page_title="Flipkart Product Recommendation", layout="centered")

# File path to the sentiment data
file_path = "Sentiment_Analysis_reviews.csv"

# Try to load sentiment data and handle potential errors
try:
    sentiment_data = pd.read_csv(file_path)
    st.write("Sentiment data loaded successfully!")
except FileNotFoundError:
    st.error("File not found. Please check the file path.")
except pd.errors.EmptyDataError:
    st.error("No data found in the file. The file seems to be empty.")
except pd.errors.ParserError:
    st.error("Error parsing the CSV file. Please check the file format.")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")

# Initialize Cohere API (replace with your actual API key)
cohere_api_key = 'ibU5EGAbyZUS3T28XYHVl5E5N802KZbcE0obURbM'  # Replace with a valid API key
co = cohere.Client(cohere_api_key)

# Function to recommend products based on user query
def recommend_products(query, sentiment_data):
    try:
        # Analyze the user's input using Cohere
        response = co.generate(prompt=query, model='command-xlarge-nightly')

        # Process response to extract keywords
        search_keywords = response.generations[0].text.strip().split()

        # Filter sentiment data based on query analysis
        recommendations = sentiment_data[sentiment_data['Review'].str.contains('|'.join(search_keywords), case=False)]

        # Sort products by sentiment score
        recommendations = recommendations.sort_values(by='Sentiment_Score', ascending=False)

        # Return top 5 products including details
        return recommendations[['Brand', 'Rating']].head(5)
    except Exception as e:
        st.error(f"An error occurred during product recommendation: {e}")
        return pd.DataFrame()

# Streamlit App Interface
st.markdown("<h1 style='text-align: center; color: #00BFFF;'> Flipkart’s Smart Product Recommender for Festival Offers</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #FFFFFF;'>Find the best-rated phones based on user reviews and sentiment analysis!</h3>", unsafe_allow_html=True)

# Add a relevant phone banner image for better UI
st.image("https://www.gizchina.com/wp-content/uploads/images/2023/08/Flagship-Phones-700x394.jpg", use_column_width=True)

# User input section with styled markdown
st.markdown("<h4 style='color: #800000;'>🚀 Tell us your preferences for the most suitable phone recommendations:</h4>", unsafe_allow_html=True)
st.markdown("<p style='color: #FAEBD7;'>For example: <b>best camera phone</b>, <b>lightweight</b>, etc.</p>", unsafe_allow_html=True)



# Create a container for input elements
with st.container():
    query = st.text_area("Describe your Needs:", height=100)
    recommend_button = st.button("🧐 view")

# Show recommendations when the button is clicked
if recommend_button and query:
    if not sentiment_data.empty:
        recommended_products = recommend_products(query, sentiment_data)
        
        if not recommended_products.empty:
            st.markdown("<h3 style='color: #32CD32;'>🎉🥳🎊  Your result:</h3>", unsafe_allow_html=True)
            
            # Display product name and rating with styled text
            for idx, row in recommended_products.iterrows():
                st.markdown(
                    f"<div style='border: 1px solid #00BFFF; border-radius: 10px; padding: 10px; margin: 10px 0; background-color: #1E1E1E;'>"
                    f"<p style='font-size:18px; color:#FFFFFF;'><b>{row['Brand']}</b> - <span style='color:#FF4500;'>{row['Rating']}</span></p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        else:
            st.warning("No matching products found. Please try a different query.")
    else:
        st.error("Sentiment data is not available. Please check the file and try again.")


