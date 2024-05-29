import uuid
from collections import defaultdict
import streamlit as st
from nlp import Pipeline, Tokenizer, StopwordRemoval, Stemmer, TextStatistics, InvertedIndex
import pandas as pd
from streamlit_option_menu import option_menu
import json
import plotly.express as px

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

# Add "Search" to the options
options = ["Document Search", "Pipeline"]
icons = ["search", "book"]

selected = option_menu(
    menu_title=None,
    options=options,
    icons=icons,
    orientation="horizontal",
)

if selected == "Document Search":
    if files:
        query = st.text_input("Enter your search query:")
        indexer = InvertedIndex()
        inverted_index, doc_pointers = indexer.process(files)
    
        if st.button('Search'):
            sorted_doc_names, sorted_scores = indexer.search(query, doc_pointers)
            
            if not sorted_doc_names:
                st.write("No documents match your search query.")
            else:
                st.write("Matching documents:")
                for rank, doc_name in enumerate(sorted_doc_names):
                    st.write(rank+1, doc_name)
                st.session_state['sorted_scores'] = sorted_scores  # Store the similarity scores in a session variable
                
        if st.button('Show Value'):
            sorted_doc_names, sorted_scores = indexer.search(query, doc_pointers)
            if 'sorted_scores' in st.session_state:
                st.write("Cosine Similarity of Matching Documents:")
                for rank, (doc_name, score) in enumerate(zip(sorted_doc_names, st.session_state['sorted_scores'])):
                    st.write(rank+1, f"{doc_name}: {score[1]}")
            else:
                st.write("No similarity scores available. Please perform a search first.")
    else:
        st.warning("Please upload files and click the submit button to continue.")


if selected == "Pipeline":
    options = ["Tokenize", "Remove Stopwords", "Stemming", "Text Statistics", "Create Inverted Index", "Search", "Document Analytics"]
    icons = ["pencil-fill", "x-circle-fill", "scissors", "bar-chart-fill", "book", "search", "bar-chart-line"]

    selected = option_menu(
        menu_title=None,
        options=options,
        icons=icons,
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
            st.download_button(
                label="Download Stemmed Text",
                data=stemmed_df.to_json(index=False),
                file_name="stemmed_text.json",
                mime="application/json",
            )

    if selected == "Text Statistics":
        if not content:
            st.warning("Please upload files and click the submit button to continue.")
        else:
            # Calculate the frequency of each word in the text
        # Create a DataFrame
            
            word_freq = stats.calc_frequency(tokenizer.tokenize(content))
            df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
            # Display the DataFrame in Streamlit
            st.dataframe(df)
            stats.tabular_format(word_freq)

            word_freq = stats.rank_words(word_freq)
            if st.button('Sorted Graph'):
                # rank_words = stats.rank_words(word_freq)
                stats.tabular_format(word_freq)
            # Plot the rank-frequency graph
            stats.freq_rank_graph(word_freq)


    if selected == "Create Inverted Index":
        if not files:
            st.warning("Please upload files and click the submit button to continue.")
        else:
            inverted_index = InvertedIndex()
            inverted_index_data, all_chunks = inverted_index.process(files)
            
            save = st.button("Visualize Indexes")

            if save:
                # Convert the inverted index to a DataFrame
                df = pd.DataFrame([(term, data["doc_count"], data["term_freq"], data["doc_ids"]) for term, data in sorted(inverted_index_data.items())], columns=['Term', 'Document Count', 'Term Frequency', 'Document IDs'])
                # Display the DataFrame as a table in Streamlit
                st.table(df)
            
            # Allow the user to download the files
            st.download_button(
                label="Download Vocabulary",
                data=json.dumps(inverted_index_data),
                file_name="vocabulary.json",
                mime="application/json",
            )
            st.download_button(
                label="Download Postings",
                data=json.dumps(all_chunks),
                file_name="postings.json",
                mime="application/json",
            )

    if selected == "Search":
        query = st.text_input("Enter your search query:")
        indexer = InvertedIndex()
        if files:
            inverted_index, doc_pointers = indexer.process(files)

        if st.button('Search'):
            sorted_doc_names, sorted_scores = indexer.search(query, doc_pointers)
            
            if not sorted_doc_names:
                st.write("No documents match your search query.")
            else:
                st.write("Matching documents:")
                for rank, doc_name in enumerate(sorted_doc_names):
                    st.write(rank+1, doc_name)
                st.session_state['sorted_scores'] = sorted_scores  # Store the similarity scores in a session variable
                
        if st.button('Show Value'):
            sorted_doc_names, sorted_scores = indexer.search(query, doc_pointers)
            if 'sorted_scores' in st.session_state:
                st.write("Cosine Similarity of Matching Documents:")
                for rank, (doc_name, score) in enumerate(zip(sorted_doc_names, st.session_state['sorted_scores'])):
                    st.write(rank+1, f"{doc_name}: {score[1]}")
            else:
                st.write("No similarity scores available. Please perform a search first.")

    if selected == "Document Analytics":
        if not files:
            st.warning("Please upload files and click the submit button to continue.")
        else:
            # Calculate the number of words in each document
            word_counts = {file.name: len(file.read().decode().split()) for file in files}
            file_names = list(word_counts.keys())
            counts = list(word_counts.values())

            # Create a bar chart
            fig = px.bar(x=file_names, y=counts, labels={'x': 'Document', 'y': 'Word Count'}, title='Word Count per Document')

            # Display the chart in Streamlit
            st.plotly_chart(fig)
