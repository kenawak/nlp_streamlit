# Streamlit NLP Application

This application is a powerful tool for Natural Language Processing (NLP) tasks. It provides various functionalities such as stopword removal, stemming, and text statistics.

## Features

1. **Stopword Removal**: This feature tokenizes the input text and removes common stopwords. The output is a list of significant words, displayed in a DataFrame and can be downloaded as a JSON file.

2. **Stemming**: This feature tokenizes the input text, removes stopwords, and applies stemming to the remaining words. Stemming is a process that reduces words to their root form. The DataFrame displays both the original and stemmed words.

3. **Text Statistics**: This feature calculates the frequency of each word in the input text and displays it in a tabular format. It also ranks the words based on their frequency and displays a rank-frequency graph. This analysis can provide insights into the most important words in a text.

## Usage

To use the application, select the desired functionality from the dropdown menu. The application will perform the selected operation on the input text and display the results.

## Screenshots

![Screenshot 1](nlp_streamlit/Screenshot 2024-05-07 053544.png)
![Screenshot 2](nlp_streamlit/Screenshot 2024-05-07 061958.png)

## Installation

To run this application locally, install the required Python packages by running `pip install -r requirements.txt` in your terminal. After installing the dependencies, run the application with the command `streamlit run app.py`.

## Getting Started

### Forking the Repository

1. On GitHub, navigate to the [repository](https://github.com/kenawak/nlp_streamlit).
2. Click **Fork** in the top-right corner of the page.

### Cloning the Repository

After forking the repository, clone it to your local machine to make changes.

1. On GitHub, navigate to **your fork** of the repository.
2. Click the **Code** button above the list of files.
3. Under "Clone with HTTPS", click the clipboard icon to copy the repository URL. If you're using an SSH key, click **Use SSH**, then click the clipboard icon.
4. Open Terminal.
5. Change the current working directory to the location where you want the cloned directory.
6. Type `git clone`, and then paste the URL you copied earlier. It will look like this:

    ```bash
    git clone https://github.com/kenawak/nlp_streamlit
    ```

7. Press **Enter** to create your local clone.

You can now make changes to the code and push them to your fork. If you'd like to propose changes to the original repository, create a pull request.

## Contact

For any questions or feedback, please contact me at [kenawakibsa95@gmail.com](mailto:kenawakibsa95@gmail.com).
