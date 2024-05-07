import streamlit as st
from nlp import Pipeline, Tokenizer, StopwordRemoval, Stemmer, TextStatistics
import pandas as pd
from streamlit_option_menu import option_menu

# Create an instance of the Pipeline class
pipeline = Pipeline()
stats = TextStatistics()
# Create a title for the app
tokenizer =  Tokenizer()
stopwords=None
stopword_remover = StopwordRemoval(None)
stemmer = Stemmer()
# text_statistics = TextStatistics()
st.title("Afan Oromo Text Preprocessing App")

#---Options for text preprocessing---


# Create a form
with st.form(key='my_form'):
    # Create a text input field inside the form
    user_input = st.text_input("Enter Afan Oromo text here:")
    # Create a submit button for the form
    file = st.file_uploader('Upload a file', type=['txt'])
    content = file.read().decode() if file else user_input
    submit_button = st.form_submit_button(label='Submit')

selected = option_menu(
    menu_title= None,
    options= ["Tokenize", "Remove Stopwords", "Stemming", "Text Statistics"],
    icons= ["pencil-fill", "bar-chart-fill"],
    orientation= "vertical",
)

if selected == "Tokenize":
    if not submit_button:
        st.warning("Please enter some text and click the submit button to continue.")
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
    tokens = tokenizer.tokenize(content)
    tokens_without_stopwords = stopword_remover.remove_stopwords(tokens)
    stemmed_words = [stemmer.stem(token) if token in tokens_without_stopwords else '' for token in tokens]

    stemmed_df = pd.DataFrame({
        'Original Words': tokens,
        'Stemmed Words': stemmed_words
    })
    st.dataframe(stemmed_df)

if selected == "Text Statistics":
    # Calculate the frequency of each word in the text
    word_freq = stats.calc_frequency(tokenizer.tokenize(content))
    stats.tabular_format(word_freq)

    # word_freq = text_stats.product_freq_rank(word_freq)

    word_freq = stats.rank_words(word_freq)

    # Plot the rank-frequency graph
    
    stats.freq_rank_graph(word_freq)