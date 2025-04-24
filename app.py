import streamlit as st
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from openai import OpenAI
import urllib.parse, requests

# Your key
client = OpenAI(api_key="sk-...")  # Replace this

# GitHub info
GITHUB_USER = "hellomaxlee"
GITHUB_REPO = "math356"
BRANCH = "main"
BASE_RAW = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{BRANCH}"

president_folders = {
    "Reagan": "President Reagan",
    "Obama": "President Obama",
    "Trump": "President Trump"
}
president_years = {
    "Reagan": range(1981, 1989),
    "Obama": range(2009, 2017),
    "Trump": range(2017, 2021)
}

@st.cache_data
def generate_files(folder, years):
    files = []
    for year in years:
        files.append(f"{folder}/State_of_the_Union_{year}.txt")
        files.append(f"{folder}/Inaugural_Address_{year}.txt")
    return files

@st.cache_data
def download_and_tokenize(file_paths):
    text = ""
    for file in file_paths:
        encoded_path = urllib.parse.quote(file)
        url = f"{BASE_RAW}/{encoded_path}"
        response = requests.get(url)
        if response.status_code == 200:
            text += response.text + " "
    return [simple_preprocess(text)]

# Load everything
tokenized_data = {p: download_and_tokenize(generate_files(f, president_years[p]))
                  for p, f in president_folders.items()}

models = {p: Word2Vec(sentences=tokenized_data[p], vector_size=50, window=5, min_count=1, sg=1, epochs=100)
          for p in tokenized_data}

def get_gpt_interpretation(president, keyword, similar_words):
    prompt = f"""
You are an expert in political speech analysis.

The user is studying how U.S. presidents use the word '{keyword}'.

For President {president}, the 5 most similar words based on Word2Vec are: {', '.join(similar_words)}.

Please explain what this suggests about the rhetorical/historical context and thematic framing of '{keyword}' in {president}'s speeches. Use 2-3 sentences and write in a helpful tone for students.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ GPT error: {e}"

# Streamlit app
st.title("🧠 Presidential Word2Vec Explorer")
st.write("Explore how Reagan, Obama, and Trump frame different concepts in their speeches using Word2Vec and GPT.")

user_word = st.text_input("Enter a word (e.g., freedom, economy, health):")

if user_word:
    for pres, model in models.items():
        st.subheader(f"🗣️ {pres}")
        if user_word in model.wv:
            similar = model.wv.most_similar(user_word, topn=5)
            words = [w for w, _ in similar]
            st.write("**Top 5 Similar Words:**", ", ".join(words))
            st.markdown("**🤖 GPT Interpretation:**")
            st.write(get_gpt_interpretation(pres, user_word, words))
        else:
            st.warning(f"'{user_word}' not found in {pres}'s vocabulary.")
