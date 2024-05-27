import re
from collections import defaultdict
import uuid
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer

# Word Tokenize
def word_tokenize(text):
    words = text.split()
    words = [word for word in words]
    return words

# Sentence Tokenize
def sentence_tokenize(text):
    sentences = re.split(r'\.', text)
    sentences = [sentence.strip() for sentence in sentences]
    return sentences

def normalize(words):
    normalized_words = []
    for word in words:
        # Replace specific characters first
        replacements = {"’": "h", "'": "h", "(": " ", ")":" "}  # Define replacements
        for original, normalized in replacements.items():
            word = word.replace(original, normalized)
        second_replacements = {" ": "",'': ''}  # Define replacements
        for original, normalized in second_replacements.items():
            word = word.replace(original, normalized)
            
        word = re.sub(r'[^\w\s]', '', word)
        word = re.sub(r'\d', '', word)
        # To handle words like MoE or EPRDF and such like that... although it is noted that case folding is the best method, we were tempted to implement this.
        upper_count = sum(1 for char in word if char.isupper())
        if upper_count >= 2:
            normalized_words.append(word)
        else:
            normalized_words.append(word.lower())
        
    # Normalize Unicode characters
    """
    nfkd:-> "Normalization Form KD". It decomposes characters by compatibility, and then recomposes them by compatibility.
    When you see "NFKD" in the context of Unicode normalization, it's referring to one of the normalization forms defined in the Unicode standard.
    """
    # normalized_text = [unicodedata.normalize('NFKD', word).encode('ASCII', 'ignore').decode('utf-8') for word in normalized_words]

    return normalized_words

class Tokenizer:
    """
    Usecage:
    >>> tokenizer = Tokenizer()
    >>> tokenizer.word_tokenize("text_content")
    >>>
    """
    def __init__(self):
        pass
    def tokenize(self, text):
        text = word_tokenize(text)
        return normalize(text)

class StopwordRemoval:
    Stopwords = {'aanee', 'agarsiisoo', 'akka', 'akkam', 'akkasumas', 'akkum', 'akkuma', 'ala', 'alatti', 'alla', 'amma', 'ammo', 'ammoo', 'an', 'ana', 'anee', 'ani', 'ati', 'bira', 'booda', 'booddee', 'dabalatees', 'dhaan', 'dudduuba', 'dugda', 'dura', 'duuba', 'eega', 'eegana', 'eegasii', 'ennaa', 'erga', 'ergii', 'f', 'faallaa', 'fagaatee', 'fakkeenyaaf', 'fi', 'fkn', 'fullee', 'fuullee', 'gajjallaa', 'gama', 'gararraa', 'garas', 'garuu', 'giddu', 'gidduu', 'gubbaa', 'ha', 'hamma', 'hanga', 'henna', 'hoggaa', 'hogguu', 'hoo', 'illee', 'immoo', 'ini', 'innaa', 'inni', 'irra', 'irraa', 'irraan', 'isa', 'isaa', 'isaaf', 'isaan', 'isaani', 'isaanii', 'isaaniitiin', 'isaanirraa', 'isaanitti', 'isaatiin', 'isarraa', 'isatti', 'isee', 'iseen', 'ishee', 'isheen', 'ishii', 'ishiif', 'ishiin', 'ishiirraa', 'ishiitti', 'isii', 'isiin', 'isin', 'isini', 'isinii', 'isiniif', 'isiniin', 'isinirraa', 'isinitti', 'ittaanee', 'itti', 'ittuu', 'itumallee', 'ituu', 'ituullee', 'jala', 'jara', 'jechaan', 'jechoota', 'jechuu', 'jechuun', 'kan', 'kana', 'kanaa', 'kanaaf', 'kanaafi', 'kanaafuu', 'kanaan', 'kanaatti', 'karaa', 'kee', 'keenna', 'keenya', 'keenyaa', 'keessa', 'keessan', 'keessatti', 'keeti', 'keetii', 'kiyya', 'koo', 'kun', 'lafa', 'lama', 'malee', 'manna', 'maqaa', 'moo', 'na', 'naa', 'naaf', 'naan', 'naannoo', 'narraa', 'natti', 'nu', 'nuhi', 'nurraa', 'nuti', 'nutti', 'nuu', 'nuuf', 'nuun', 'nuy', 'nuyi', 'odoo', 'ofii', 'oggaa', 'ol', 'oliif', 'oliin', 'oo', 'osoo', 'otoo', 'otumallee', 'otuu', 'otuullee', 'saaniif', 'sadii', 'sana', 'saniif', 'si', 'sii', 'siif', 'siin', 'silaa', 'simmoo', 'sinitti', 'siqee', 'sirraa', 'sitti', 'sun', 'ta`ullee', 'tahullee', 'tahuyyu', 'tahuyyuu', 'tana', 'tanaaf', 'tanaafi', 'tanaafuu', 'tawullee', 'ta‟ullee', 'teenya', 'teessan', 'tiyya', 'too', 'tti', 'tun', 'utuu', 'waahee', 'waan', 'waggaa', 'wajjin', 'warra', 'woo', 'yammuu', 'yemmuu', 'yeroo', 'ykn', 'yommii', 'yommuu', 'yoo', 'yookaan', 'yookiin', 'yookinimoo', 'yoolinimoo', 'yoom'}
    # Iniatialize
    """
    Usecage:
    >>> StopwordRemoval = StopwordRemoval()
    >>> StopwordRemoval.remove_stopwords("word_tokens")
    >>>
    """
    def __init__(self, stopwords=None):
        if not stopwords:
            stopwords = self.Stopwords
        self.stopwords = set(stopwords)

    def remove_stopwords(self, tokens: iter) -> list:
        filtered_tokens = [
            token for token in tokens if token.lower() not in self.stopwords]
        return filtered_tokens
    def get_stopwords(self, tokens: iter) -> list:
        stopwords = [token for token in tokens if token.lower() in self.stopwords]
        freq_words = [[word, stopwords.count(word)] for word in set(stopwords)]
        return freq_words
    def get_graph(self, tokens: iter) -> None:
        non_stopwords = self.remove_stopwords(tokens)
        stopwords = self.get_stopwords(tokens)
        total_words = len(tokens)

        label = ["Total Words", "Stopwords", "Non-Stopwords"]
        source = [0, 0]  # Both Stopwords and Non-Stopwords originate from Total Words
        target = [1, 2]  # Targets are Stopwords and Non-Stopwords
        value = [len(stopwords), len(non_stopwords)]  # The values are the counts of Stopwords and Non-Stopwords

        link = dict(source=source, target=target, value=value)
        node = dict(label=label, pad=50, thickness=5)
        data = go.Sankey(link=link, node=node)

        fig = go.Figure(data)
        st.plotly_chart(fig, use_container_width=True)
        
class Stemmer:
    """
    Usage:
    
    >>> stemmer = Stemmer()
    >>> stemmer.rule_cluster_7("gaggabaabaa")) 
    
    Or
    
    >>> stemmer.stem(word_token) 
    
    If we have list of tokenized words
    
    >>> stemmer.stem(word_token) for word_token in word_tokens 
    
    """
    def __init__(self):
        pass
    def apply_cluster_rules(self, word):
        # Define rule clusters
        clusters = [
            self.rule_cluster_1,
            self.rule_cluster_2,
            self.rule_cluster_3,
            self.rule_cluster_4,
            self.rule_cluster_5,
            self.rule_cluster_6,
            self.rule_cluster_7,
        ]
        # More rules (morphological rules) can be added to enhance it more
            
        # Apply rules from each cluster
        for cluster in clusters:
            stemmed_word = cluster(word)
            if stemmed_word:
                return stemmed_word
        # If no rule matches, return the original word
        return word
    
    def measure(self, word):
        # Calculate the number of vowel-consonant sequences in the word
        # A vowel-consonant sequence is defined where consecutive vowels or consonants are counted as one
        vowels = set("aeiou")
        measure = 0
        consecutive_count = 0
        for char in word:
            if char.lower() in vowels:
                if consecutive_count == 0:
                    measure += 1
                    consecutive_count = 1
            else:
                consecutive_count = 0
        return measure
    def stem(self, word):
        stemmed_word = self.apply_cluster_rules(word)
        return stemmed_word
    def rule_cluster_3(self, word):
        # Rule cluster 3: Delete suffixes if measure >= 1 and ends with consonant
        # Example: "eenya", "ina", "offaa", "annoo", "umsa", "ummaa", "insa"
        suffixes = ["eenya", "ina", "offaa", "annoo", "umsa", "ummaa", "insa", "iinsa","am","ni","affaa"]
        for suffix in suffixes:
            if word.endswith(suffix):
                if self.measure(word) >= 1:
                    if word[:-len(suffix)] and word[:-len(suffix)][-1] in "'bcdfghjklmnpqrstvwxyz":
                        return word[:-len(suffix)]
        return None
    def rule_cluster_4(self, word):
        # Rule cluster 5: Delete suffixes if measure >= 1, or substitute with "`" if measure = 0
        # Example: "`aa", "'e", "'u", "'ee", "suu", "sa", "sse", "nya"
        suffixes = ["`aa", "'e", "'u", "'ee", "suu","ssa", "sa", "sse", "se", "si", "nye", "nya"]
        for suffix in suffixes:
            if word.endswith(suffix):
                if self.measure(word) >= 1:
                    return word[:-len(suffix)]
                elif self.measure(word) == 0:
                    return word[:-len(suffix)] + "`"
        return None
        
    def rule_cluster_5(self, word):
        # Rule cluster 4: Special cases
        # Example: "du", "di", "dan", "wwan"
        special_cases = {
        "du":"to",
        "di":"to",
        "dan":"to",
        "lee":"la",
        "wwan":"sa",
        "een":"af",
        "an":"af",
        "f":"sha",
        "n":"sha"
    }
        
        for suffix, condition in special_cases.items():
            if word.endswith(suffix):
                # print(condition)
                if condition == "to":
                    # print(condition)
                    preceding_chars = word[:-len(suffix)]
                    if preceding_chars.endswith(("b" , "g" , "d")):
                        if self.measure(word) >= 1:
                            return word[:-len(suffix)]
                        elif self.measure(word)==0:
                            return word[:-len(suffix)] + "d"
                elif suffix == "leeee" and self.measure(word) >= 1:
                    return word[:-len(suffix)]
                elif suffix in ["wwan", "een"] and self.measure(word) >= 1:
                    return word[:-len(suffix)]
                elif suffix == "an" and self.measure(word) >= 1:
                    return word[:-len(suffix)]
                elif suffix in ["f", "n"] and self.measure(word) >= 1:
                    return word[:-len(suffix)]
        return None
    

    def rule_cluster_7(self, word):
        if not word.startswith(("a", "e", "i", "o", "u")):
            if word[:2] == word[3:5]:
                return word[3:]
            elif word[:2] == word[2:5]:
                return word[2:]
        if word.startswith(("a", "e", "i", "o", "u")):
            if word[:2] == word[2:4] or "'" in word or "h" in word:
                word = word.replace("'", word[3:4])
                word = word.replace("h", word[3:4])
                if word[:2] == word[2:4]:
                    return word[2:]
        return None

    def rule_cluster_1(self, word):
        suffixes = ["olee","olii", "oolii" , "oota", "ota", "oolee" , "icha","ichi","oma","fis","siis","ooma","siif", "fam","ata"]
        for suffix in suffixes:
            if word.endswith(suffix):
                if self.measure(word) >= 1:
                    return word[:-len(suffix)]
    def rule_cluster_2(self, word):
        suffixes = ["ittii", "dha", "ttii", "irra", "tti","tii", "rra"]
        for suffix in suffixes:
            if self.measure(word) >= 1:
                if word.endswith(suffix):
                    vv = ['aa', 'ae', 'ai', 'ao', 'au', 'ea', 'ee', 'ei', 'eo', 'eu', 'ia', 'ie', 'ii', 'io', 'iu', 'oa', 'oe', 'oi', 'oo', 'ou', 'ua', 'ue', 'ui', 'uo', 'uu']
                    for end in vv:
                        if word[:-len(suffix)].endswith(end):
                            return word[:-len(suffix)]
                    if word[:-len(suffix)].endswith("ti"):
                        return word[:-len(suffix)]
                    else:
                        return word[:-len(suffix)]
        return None
    def rule_cluster_6(self, word):
        suffixes = ["te", "tu", "ti", "tee", "tuu", "ne", "nu", "na", "nne", "nnu", "nna", "dhaa", "chaaf",
                    "dhaaf", "tiif", "ach", "adh", "chuu", "at", "att", "ch", "tanu", "tanuu", "tan", "tani"]
        for suffix in suffixes:
            if word.endswith(suffix):
                if self.measure(word) >= 1:
                    return word[:-len(suffix)]
                else:
                    return word[:-len(suffix)] + "t"
        return None


class TextStatistics:    
    def __init__(self):
        self.tokenizer = Tokenizer()

    def apply_statistics(self, words):
        # Define rule clusters
        stats = [
            self.tabular_format,
            
        ]
                
        # Apply rules from each cluster
        for stat in stats:
            words = stat(words)
        return words
    def get_statistics(self, text):
        words = self.tokenizer.tokenize(text)
        words = self.calc_frequency(words)
        words = self.apply_statistics(words)
        return words
    def calc_frequency(self, words):
        freq_words = [[word, words.count(word)] for word in set(words)]
        return freq_words
    
    # Ranking words according to their frequency, and giving appending their corresponding ranks
    def rank_words(self, words):
        words.sort(key=lambda x: x[1], reverse=True)
        for i in range(0, len(words)):
            words[i].append(i+1)
        return words

    # Calculate the product of rank and frequency; the function takes in the ranked list of the words and their corresponding frequency also
    def product_freq_rank(self, words):
        for i in range(0, len(words)):
            product_rank_freq = words[i][1]*words[i][2]
            words[i].append(product_rank_freq)
        return words

    def tabular_format(self, words):
        # Create a DataFrame
        df = pd.DataFrame(words, columns=['Word', 'Frequency'])
        # Display the DataFrame in Streamlit
        st.dataframe(df)

        # Bar chart for word frequencies
        words, frequencies = zip(*[[pair[0], pair[1]] for pair in words])
        df = pd.DataFrame({'words': words, 'frequencies': frequencies})

        fig = go.Figure(data=[
            go.Bar(name='Words', x=df['words'], y=df['frequencies'], marker_color='indianred'),
        ])

        fig.update_layout(
            title='Word Frequencies',
            xaxis=dict(
                title='Words',
                gridcolor='white',
                gridwidth=2,
            ),
            yaxis=dict(
                title='Frequencies',
                gridcolor='white',
                gridwidth=2,
            ),
            paper_bgcolor='rgb(0, 0, 0)',
            plot_bgcolor='rgb(0, 0, 0)',
        )

        st.plotly_chart(fig)

    def freq_rank_graph(self, words):
        ranks = [pair[2] for pair in words]
        frequencies = [pair[1] for pair in words]

        # Log-log plot for rank-frequency graph
        df = pd.DataFrame({'ranks': ranks, 'frequencies': frequencies})

        fig = go.Figure(data=[
            go.Scatter(name='Rank-Frequency', x=df['ranks'], y=df['frequencies'], mode='lines+markers'),
        ])

        fig.update_layout(
            title='Rank-Frequency Graph',
            xaxis=dict(
                title='Ranks',
                type='log',
                gridcolor='white',
                gridwidth=2,
            ),
            yaxis=dict(
                title='Frequencies',
                type='log',
                gridcolor='white',
                gridwidth=2,
            ),
            paper_bgcolor='rgb(0, 0, 0)',
            plot_bgcolor='rgb(0, 0, 0)',
        )

        st.plotly_chart(fig)
class Pipeline:
    """
    Text Preprocessing Pipeline
    
    This class integrates the application of varios preprocessing steps on Afaan Oromoo text,

    Methods:
        - preprocess(text): Applies the preprocessing steps to the input text.

    USAGE:
     # Step 1: Instantiate the Pipeline
     # Step 2: Preprocess the Content
     # Step 3: Print or use the preprocessed content
     Example:
     >>> pipeline = Pipeline()
     >>> preprocessed_text = pipeline.preprocess("Beeksisa MoEdhaan ba'een.")
     >>> print(preprocessed_text)

     
    """
    def __init__(self, stopwords=None):
        """
        Initializing the preprocessing pipeline
        stopwords -> A set of stopwords; if not given default stopwords will be used.
        
        """
        self.tokenizer =  Tokenizer()
        self.stopword_remover = StopwordRemoval(stopwords)
        self.stemmer = Stemmer()
        self.text_statistics = TextStatistics()
    def preprocess(self, text: str):
        """
        Apply the preprocessing steps to the input text.

        Parameters:
        - text (str): The input text to be prepocessed

        Returns:
         - str: The preprocessed text.
        """
        
        # Step 1: Tokenize
        tokens = self.tokenizer.tokenize(text)
        # print(tokens)
        
        # Step 2: Remove stopwrods
        tokens_without_stopwords = self.stopword_remover.remove_stopwords(tokens)
        

        # Step 3:
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens_without_stopwords]

        # Step 4: Join back and return
        
        preprocessed_text = stemmed_tokens
        return preprocessed_text
        
def calculate_term_frequency(document):
    # Calculate the term frequency of each term in the document
    term_frequency = {}
    for term in document:
        term_frequency[term] = term_frequency.get(term, 0) + 1
    return term_frequency
def create_inverted_index(files):
    pipeline = Pipeline()
    
    inverted_index = defaultdict(lambda: {"doc_count": 0, "term_freq": 0, "doc_ids": set(), "term_freq_doc": {}})
    all_chunks = {}

    document_vectors = []

    # Iterate over all uploaded files
    for file in files:
        document = file.read().decode()
        doc_id = str(uuid.uuid4())
        preprocessed_document = pipeline.preprocess(document)
        term_frequency = calculate_term_frequency(preprocessed_document)
        
        # Update inverted index and document vectors
        for term in preprocessed_document:
            if term == "":
                continue
            inverted_index[term]["doc_ids"].add(doc_id)  # Use a set to avoid duplicates
            inverted_index[term]["term_freq"] += 1  # Increment term frequency across all documents
            
            # Update term frequency for this document
            inverted_index[term]["term_freq_doc"][doc_id] = term_frequency.get(term, 0)
        
        file_name = file.name  # Extract the file name from the path
        all_chunks[doc_id] = {"document_id": doc_id, "file_name": file_name, "content": document}

    # Convert sets to lists for JSON serialization
    for term in inverted_index:
        inverted_index[term]["doc_count"] = len(inverted_index[term]["doc_ids"])
        inverted_index[term]["doc_ids"] = list(inverted_index[term]["doc_ids"])

    return inverted_index, all_chunks

def calculate_document_vectors(documents):
    pipeline = Pipeline()
    preprocessed_docs = [' '.join(pipeline.preprocess(doc.get("content", ""))) for doc in documents.values()]
    
    vectorizer = CountVectorizer()
    document_vectors = vectorizer.fit_transform(preprocessed_docs)
    
    return document_vectors, vectorizer
def manual_cosine_similarity(doc_vector, query_vector):
    dot_product = sum(a * b for a, b in zip(doc_vector, query_vector))
    magnitude_doc = sum(a * a for a in doc_vector) ** 0.5
    magnitude_query = sum(b * b for b in query_vector) ** 0.5
    if magnitude_doc == 0 or magnitude_query == 0:
        return 0.0
    return dot_product / (magnitude_doc * magnitude_query)

def retrieve_documents(query, documents, vectorizer, document_vectors):
    pipeline = Pipeline()
    preprocessed_query = ' '.join(pipeline.preprocess(query))
    query_vector = vectorizer.transform([preprocessed_query]).toarray()[0]  # Convert to dense array and ensure it's 1D

    similarity_scores = []
    for doc_id, doc_vector in zip(documents.keys(), document_vectors):
        doc_vector = doc_vector.toarray()[0]  # Convert each document vector to a dense 1D array
        similarity = manual_cosine_similarity(doc_vector, query_vector)
        similarity_scores.append((doc_id, similarity))  # Store tuples of (doc_id, similarity)

    sorted_similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    sorted_documents = [doc_id for doc_id, _ in sorted_similarity_scores]
    return sorted_documents, sorted_similarity_scores

def display_similarity_scores(sorted_similarity_scores, document_names):
    print("Cosine Similarity Scores:")
    for doc_id, similarity in sorted_similarity_scores:
        print(f"{document_names[doc_id]}: {similarity:.4f}")

class InvertedIndex:
    def __init__(self):
        pass
    
    def process(self, files):
        return create_inverted_index(files)
    def search(self, query, doc_pointers):
        documents = doc_pointers
        document_vectors, vectorizer = calculate_document_vectors(doc_pointers)
        sorted_doc_ids, sorted_similarity_scores = retrieve_documents(query, documents, vectorizer, document_vectors)
        sorted_documents = [documents[doc_id]["file_name"] for doc_id in sorted_doc_ids]
        return sorted_documents, sorted_similarity_scores
