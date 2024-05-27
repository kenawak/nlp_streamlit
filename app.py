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

# Add "Search" to the options
options = ["Tokenize", "Remove Stopwords", "Stemming", "Text Statistics", "Create Inverted Index", "Search"]
icons = ["pencil-fill", "x-circle-fill", "scissors", "bar-chart-fill", "book", "search"]

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
        word_freq = stats.calc_frequency(tokenizer.tokenize(content))
        stats.tabular_format(word_freq)

            word_freq = stats.rank_words(word_freq)

        # Plot the rank-frequency graph
        stats.freq_rank_graph(word_freq)

if selected == "Create Inverted Index":
    if not content:
        st.warning("Please upload files and click the submit button to continue.")
    else:
        # Create the inverted index from the uploaded files
        # content = pipeline.preprocess(content)
        inverted_index_data, all_chunks = inverted_index.process(files)
        
        save = st.button("Visualize Indexes")
        # Create vocabulary file

        vocab_file_path = 'vocabulary.json'
        vocab_data = {term: {"doc_count": data["doc_count"], "term_freq": data["term_freq"], "doc_ids": data["doc_ids"]} for term, data in sorted(inverted_index_data.items())}
        # Create postings file
        postings_file_path = 'postings.json'
        postings_data = {chunk["file_name"]: {"document_id": doc_id, "file_name": chunk["file_name"]} for doc_id, chunk in all_chunks.items()}

        if save:
            # Convert the inverted index to a DataFrame
            df = pd.DataFrame([(term, data["doc_count"], data["term_freq"], data["doc_ids"]) for term, data in sorted(inverted_index_data.items())], columns=['Term', 'Document Count', 'Term Frequency', 'Document IDs'])
            # Display the DataFrame as a table in Streamlit
            st.table(df)
            # Convert the inverted index to a DataFrame
            # df = pd.DataFrame(list(inverted_index.items()), columns=['Term', 'Documents'])

            # # Display the DataFrame as a table in Streamlit
            # st.table(df)
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

if selected == "Search":
    # Load the inverted index and postings from the JSON files
    with open('vocabulary.json', 'r') as vocab_file:
        inverted_index_data = json.load(vocab_file)
    with open('postings.json', 'r') as postings_file:
        postings_data = json.load(postings_file)

    # Get the search query from the user
    query = st.text_input("Enter your search query:")

    if st.button('Search'):
        # Tokenize the search query
        query_terms = tokenizer.tokenize(query)

        # Get the list of document IDs for each term in the query
        doc_ids_per_term = [set(inverted_index_data[term]['doc_ids']) for term in query_terms if term in inverted_index_data]

        if not doc_ids_per_term:
            st.write("No documents match your search query.")
        else:
            # Find the intersection of the document ID lists
            common_doc_ids = set.intersection(*doc_ids_per_term)

            st.write(f"Common document IDs: {common_doc_ids}")
            st.write(f"Postings data: {postings_data}")

            # Retrieve the file names of the matching documents
            # Create a dictionary that maps document IDs to file names
            doc_id_to_file_name = {v['document_id']: k for k, v in postings_data.items()}

            # Retrieve the file names of the matching documents
            matching_files = [doc_id_to_file_name[str(doc_id)] for doc_id in common_doc_ids if str(doc_id) in doc_id_to_file_name]

            st.write("Matching documents:")
            for file_name in matching_files:
                st.write(file_name)
