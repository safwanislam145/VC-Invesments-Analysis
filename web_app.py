import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jaccard
import matplotlib.pyplot as plt
import seaborn as sns
import time  # For simulating progress

# Add a title and description
st.title("Company Similarity Checker")
st.markdown("""
Welcome to the Company Similarity Checker! 🚀  
This app calculates the similarity between companies based on various attributes, such as:
- **Categories**
- **Descriptions**
- **Employee Counts**

Please wait while we preprocess the data. This might take a few moments.  
""")

# Show a progress bar while preprocessing starts
progress_bar = st.progress(0)

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to tokenize and lemmatize text
def preprocess_text(text):
    # Tokenize text
    tokens = word_tokenize(text.lower())
    # Remove stopwords
    clean_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    # Lemmatize text
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in clean_tokens]
    return ' '.join(lemmatized_tokens)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_data.csv")

data = load_data()

# Preprocess the data
@st.cache_data
def preprocess_data(data):
    # Preprocess the unified description
    data['cleaned_description'] = data['Unified Description'].dropna().apply(preprocess_text)
    
    # Split categories into lowercase exploded rows
    data['Top Level Category'] = data['Top Level Category'].str.lower()
    data['Top Level Category'] = data['Top Level Category'].fillna('Unknown')
    split_category = data['Top Level Category'].str.split(';')
    
    # Explode lists and reindex to avoid duplicate labels
    exploded_category = split_category.explode().reset_index(drop=True)
    data['Exploded Category'] = exploded_category

    # Normalize Employee Count
    scaler = MinMaxScaler()
    data['Employee Count Scaled'] = scaler.fit_transform(data[['Employee Count']].fillna(0))
    
    # TF-IDF for descriptions
    tfidf = TfidfVectorizer(stop_words='english')
    descriptions_tfidf = tfidf.fit_transform(data['cleaned_description'].fillna(''))
    return data, tfidf, descriptions_tfidf

data, tfidf, descriptions_tfidf = preprocess_data(data)

# After preprocessing
st.markdown("---")
st.header("Ready to Compare Companies")
st.markdown("Use the sidebar to enter a company name and explore similarities.") 

def compute_similarity(target_company, group):
    results = []
    group_list = group.reset_index().to_dict('records')  # Convert DataFrame to list of dictionaries
    target_index = target_company.name  # Get the index of the target company

    for company in group_list:
        # Categories Similarity (Jaccard Similarity using sets)
        categories_a = set(target_company['Top Level Category'].split('; ') + target_company['Secondary Category'].split('; '))
        categories_b = set(company['Top Level Category'].split('; ') + company['Secondary Category'].split('; '))
        
        # Check for 'Unknown' categories
        if 'Unknown' in categories_a or 'Unknown' in categories_b:
            category_similarity = 0  # Default to 0 if any category is 'Unknown'
            weight = 0  # Weight is 0 if any category is 'Unknown'
        else:
            category_similarity = len(categories_a.intersection(categories_b)) / len(categories_a.union(categories_b))
            weight = 0.4  # Standard weight when categories are valid

        # Description Similarity (Cosine Similarity)
        desc_a = descriptions_tfidf[target_index].toarray()  # Access vector by DataFrame index
        desc_b = descriptions_tfidf[company['index']].toarray()  # Access vector by DataFrame index
        desc_similarity = cosine_similarity(desc_a, desc_b).item()

        # Improved Employee Count Similarity
        emp_a = target_company['Employee Count']
        emp_b = company['Employee Count']
        if emp_a > 0 and emp_b > 0:
            employee_similarity = 1 - abs(np.log1p(emp_a) - np.log1p(emp_b)) / max(np.log1p(emp_a), np.log1p(emp_b))
        else:
            employee_similarity = 0  # Set to 0 if employee counts are invalid or 0

        # Composite Similarity
        composite_similarity = (category_similarity * weight +
                                 desc_similarity * 0.4 +
                                 employee_similarity * 0.2)

        results.append({
            'Compared Company': company['Name'],
            'Category Similarity': category_similarity,
            'Description Similarity': desc_similarity,
            'Employee Similarity': employee_similarity,
            'Composite Similarity': composite_similarity
        })
    
    return pd.DataFrame(results)

# Main page section to view companies by category
st.markdown("---")
st.header("View Companies by Category")

# Get unique categories for filtering
unique_categories = data['Top Level Category'].dropna().unique()

# Dropdown for selecting a category
selected_category = st.selectbox("Select a Category to View Companies:", unique_categories)

# Filter and display companies in the selected category
if selected_category:
    st.subheader(f"Companies in Selected Category: {selected_category.capitalize()}")
    filtered_companies = data[data['Top Level Category'] == selected_category]
    st.dataframe(filtered_companies[['Name', 'Top Level Category', 'Secondary Category', 'Employee Count']])
    
    
## Sidebar input
st.sidebar.header("Input")

# Sidebar input for company name
company_name = st.sidebar.text_input("Enter a Company Name:", key="company_name_input")

if company_name and company_name in data['Name'].values:
    target_company = data[data['Name'] == company_name].iloc[0]
    top_level_category = target_company['Top Level Category']
    st.title(f"Results for {company_name}")
    st.write(f"Target Company: {target_company['Name']}")
    st.write(f"Top Level Category: {top_level_category}")

    # Filter companies with the same Top Level Category
    same_category_companies = data[data['Top Level Category'] == top_level_category]

    # Perform initial comparison or refresh
    if "random_companies" not in st.session_state:
        st.session_state.random_companies = same_category_companies.sample(n=min(100, len(same_category_companies)), random_state=42)

    if st.button("Refresh Companies"):
        st.session_state.random_companies = same_category_companies.sample(n=min(100, len(same_category_companies)))

    similarity_df = compute_similarity(target_company, st.session_state.random_companies)

    # Display results
    st.subheader(f"Similarity Scores for {company_name}")
    st.dataframe(similarity_df)

    # Top Similar Companies
    top_similarities = similarity_df.sort_values(by='Composite Similarity', ascending=False).head(5)
    st.markdown("---")
    st.subheader("Top 5 Most Similar Companies")
    st.dataframe(top_similarities)

    # Plot top similarities
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top_similarities, x='Composite Similarity', y='Compared Company', ax=ax)
    ax.set_title(f"Top 5 Similarity Scores for {company_name}")
    st.pyplot(fig)

    # Display all companies in the same category
    st.markdown("---")
    st.subheader(f"All Companies in Top Level Category: {top_level_category.capitalize()}")
    st.dataframe(same_category_companies[['Name', 'Top Level Category', 'Secondary Category', 'Employee Count']])
else:
    st.warning("Please enter a valid company name to see its similarity results.")