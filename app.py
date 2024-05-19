import uuid
from collections import defaultdict
import streamlit as st
from nlp import Pipeline, Tokenizer, StopwordRemoval, Stemmer, TextStatistics, InvertedIndex
import pandas as pd
from streamlit_option_menu import option_menu
import json

# Create an instance of the Pipeline class
pipeline = Pipeline()
stats = TextStatistics()
tokenizer = Tokenizer()
stopword_remover = StopwordRemoval(None)
stemmer = Stemmer()
inverted_index = InvertedIndex()

# Title of the app
st.title("Afan Oromo Text Preprocessing App")


# Create a form
if 'content' not in st.session_state:
    st.session_state['content'] = ""
if 'file_names' not in st.session_state:
    st.session_state['file_names'] = []

with st.form(key='my_form'):
    # Create a file uploader for multiple files
    files = st.file_uploader("Choose files", accept_multiple_files=True, type=['txt'])
    submitted = st.form_submit_button("Submit")

if submitted and files:
    # Read the content of all uploaded files
    file_names = []
    content = []
    for file in files:
        file_content = file.read().decode()
        content.append(file_content)
        file_names.append(file.name)
    st.session_state['content'] = "\n".join(content)
    st.session_state['file_names'] = file_names
selected = option_menu(
    menu_title=None,
    options=["Tokenize", "Remove Stopwords", "Stemming", "Text Statistics", "Create Inverted Index"],
    icons=["pencil-fill", "x-circle-fill", "scissors", "bar-chart-fill", "book"],
    orientation="vertical",
)
content = st.session_state['content']
file_names = st.session_state['file_names']

if selected == "Tokenize":
    if not content:
        st.warning("Please upload files and click the submit button to continue.")
    else:
        tokenized_text = tokenizer.tokenize(content)
        st.write(f"Number of tokens: {len(tokenized_text)}")
        data = {'word': [], 'characters': []}
        for word in tokenized_text:
            data['word'].append(word)
            data['characters'].append(len(word))
        df = pd.DataFrame(data)
        st.dataframe(df)

        st.download_button(
            label="Download Tokenized Text",
            data=df.to_json(index=False),
            file_name="tokenized_text.json",
            mime="application/json",
        )

if selected == "Remove Stopwords":
    if not content:
        st.warning("Please upload files and click the submit button to continue.")
    else:
        removed_stopwords = stopword_remover.get_stopwords(tokenizer.tokenize(content))
        
        stopwords_df = pd.DataFrame(removed_stopwords, columns=['Stopwords', 'Frequency'])
        st.dataframe(stopwords_df)
        graph = stopword_remover.get_graph(tokenizer.tokenize(content))
        st.download_button(
            label="Download Removed Stopwords",
            data=stopwords_df.to_json(index=False).encode(),
            file_name="removed_stopwords.json",
            mime="application/json",
        )

if selected == "Stemming":
    if not content:
        st.warning("Please upload files and click the submit button to continue.")
    else:
        tokens = tokenizer.tokenize(content)
        tokens_without_stopwords = stopword_remover.remove_stopwords(tokens)
        stemmed_words = [stemmer.stem(token) if token in tokens_without_stopwords else '' for token in tokens]

        stemmed_df = pd.DataFrame({
            'Original Words': tokens,
            'Stemmed Words': stemmed_words
        })
        st.dataframe(stemmed_df)

if selected == "Text Statistics":
    if not content:
        st.warning("Please upload files and click the submit button to continue.")
    else:
        # Calculate the frequency of each word in the text
        word_freq = stats.calc_frequency(tokenizer.tokenize(content))
        stats.tabular_format(word_freq)

        word_freq = stats.rank_words(word_freq)

        # Plot the rank-frequency graph
        stats.freq_rank_graph(word_freq)

def create_inverted_index_from_content(contents, file_names):
    inverted_index = defaultdict(lambda: {"doc_count": 0, "term_freq": 0, "doc_ids": set()})
    all_chunks = {}

    for content, file_name in zip(contents, file_names):
        doc_id = str(uuid.uuid4())
        preprocessed_document = pipeline.preprocess(content)
        
        for term in preprocessed_document:
            if term == "":
                continue
            inverted_index[term]["doc_ids"].add(doc_id)
            inverted_index[term]["term_freq"] += 1
        
        all_chunks[doc_id] = {"document_id": doc_id, "file_name": file_name}

    for term in inverted_index:
        inverted_index[term]["doc_count"] = len(inverted_index[term]["doc_ids"])
        inverted_index[term]["doc_ids"] = list(inverted_index[term]["doc_ids"])

    return inverted_index, all_chunks

if selected == "Create Inverted Index":
    if not content:
        st.warning("Please upload files and click the submit button to continue.")
    else:
        # Create the inverted index from the uploaded files
        inverted_index_data, all_chunks = create_inverted_index_from_content(content.split('\n'), file_names)
        
        # Save the vocabulary in JSON format
        vocab_file_path = 'vocabulary.json'
        vocab_data = {term: data for term, data in sorted(inverted_index_data.items())}
        with open(vocab_file_path, 'w') as vocab_file:
            json.dump(vocab_data, vocab_file, indent=4)
        st.success(f"Vocabulary saved to {vocab_file_path}")

        # Save the postings in JSON format
        postings_file_path = 'postings.json'
        postings_data = {doc_id: data for doc_id, data in all_chunks.items()}
        with open(postings_file_path, 'w') as postings_file:
            json.dump(postings_data, postings_file, indent=4)
        st.success(f"Postings saved to {postings_file_path}")

        # Allow the user to download the files
        st.download_button(
            label="Download Vocabulary",
            data=json.dumps(vocab_data),
            file_name="vocabulary.json",
            mime="application/json",
        )
        st.download_button(
            label="Download Postings",
            data=json.dumps(postings_data),
            file_name="postings.json",
            mime="application/json",
        )
