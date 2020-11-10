
## FE 595 Midterm

\
\
**Summary**
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
A collaborative project between group members Yuwen Jin, Minghao Kang, Fangchi Wu, and Shiraz Bheda. The goal is to create a web interface from which the user may select up to 8 unique NLP analytical tools to be applied on a text string that is entered into a blank text box by the user:

[![Screenshot-2020-11-09-210040.png](https://i.postimg.cc/Ss0w8FNk/Screenshot-2020-11-09-210040.png)](https://postimg.cc/VSRGQhCV)

\
\
**Description of NLP tools**
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
1. Sentiment Analysis: indicates whether sentiment in the text string is positive or negative
2. Language Detect: returns the name of the most possible language that the string is written in
3. Tokenize: tokenizes the string
4. Top Ten: returns a list of up to ten of the most frequently used words
5. Part of Speech Tagging: assigns each word in the string to a part of speech
6. English to French: translates the string from English to French
7. English to Chinese: translates the string from English to Chinese
8. Spell Check: returns spell check result of the text

\
\
**Instructions**
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
One way to **access the website** is to run 'flask api.py' in this project directly. In this case, if you're using Windows system the homepage address is '127.0.0.1:5000', and if you're using ios system the address is '0.0.0.0:8000'. 
The other way is to launch your AWS instance and connect to your running instance via your terminal. After installing git and python3, and using git clone to clone the repository to your terminal,you will need to install the requirements of the flask api prior to running the script itself.

**Open the home page**, enter text in the blank text box and select method(s) via check box, click 'submit' button then you'll see the result page. 
If the text entry is blank you'll receive a message saying 'your text is empty', while if you click 'submit' without choosing any analyzing method you'll get another message saying 'please choose at least one analyzing method'. The 'back' button can take you back to the previous page.

NOTE: While most of the required packages to run this script have been placed on a 'requirements.txt' page, there is a subfolder titled 'materials' that contains several additional packages that may need to be manually installed by accessing the file path directory. Once this has been completed, you can uncomment lines in 'flask api.py', 'home.html' and 'result.html', then you'll see another NLP toy, an AI response machine.

\
\
**Example output**
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
[![Screenshot-2020-11-09-213940.png](https://i.postimg.cc/1XKdF8fZ/Screenshot-2020-11-09-213940.png)](https://postimg.cc/sGvKdDYn)

