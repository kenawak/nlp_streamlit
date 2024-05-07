# Streamlit NLP Application

This application provides various Natural Language Processing (NLP) functionalities, including stopword removal, stemming, and text statistics.

## Features

1. **Stopword Removal**: Tokenizes the input content and removes common stopwords. The resulting words are displayed in a DataFrame and can be downloaded as a JSON file.

2. **Stemming**: Tokenizes the input content, removes stopwords, and then applies stemming to the remaining words. Stemming is a process that reduces words to their root form. The original and stemmed words are displayed in a DataFrame.

3. **Text Statistics**: Calculates the frequency of each word in the input content and displays it in a tabular format. It also ranks the words based on their frequency and displays a rank-frequency graph. This is a common analysis in NLP and can provide insights into the most important words in a text.

## Usage

Select the desired functionality from the dropdown menu. The application will perform the selected operation on the input content and display the results.

## Screenshots

![Screenshot 1](<Screenshot 2024-05-07 053544.png>)
![Screenshot 2](<Screenshot 2024-05-07 061958.png>)

## Installation

To run this application locally, you'll need to install the required Python packages. You can do this by running `pip install -r requirements.txt` in your terminal.

After installing the dependencies, you can run the application with the command `streamlit run app.py`.

## Contact

If you have any questions or feedback, please feel free to contact me at <your_email_address>.
