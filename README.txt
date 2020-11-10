# FE595Midterm

Summary
-------

A collaborative project between group members Yuwen Jin, Minghao Kang, Fangchi Wu, and Shiraz Bheda. The goal is to create a web interface from which the user may select up to 8 unique NLP analytical tools to be applied on a text string that can be entered into a blank text box by the user:

[![Screenshot-2020-11-09-210040.png](https://i.postimg.cc/Ss0w8FNk/Screenshot-2020-11-09-210040.png)](https://postimg.cc/VSRGQhCV)


Description of NLP tools
------------------------
1. Sentiment Analysis: indicates whether sentiment in the text string is positive or negative
2. Language Detect: returns the name of the language that the string is written in
3. Tokenize: tokenizes the string
4. Top Ten: returns a list of up to ten of the most frequently used words
5. Part of Speech Tagging: assigns each word in the string to a part of speech
6. English to French: translates the string from English to French
7. English to Chinese: translates the string from English to Chinese
8. Spell Check: Returns spelling errors (if any)


Instructions
------------
Launch your AWS instance and connect to your running instance via your terminal.
After installing git and python3, and using git clone to clone the repository to your terminal,you will need to install the requirements of the flask api prior to running the script itself.

NOTE: While most of the required packages to run this script have been placed on a 'requirements.txt' page, there is a subfolder titled 'materials' that contains several additional packages that must be manually installed by accessing the file path directory. Once this has been completed, the flask api.py file will run successfully. 


Example output
--------------

[![Screenshot-2020-11-09-213940.png](https://i.postimg.cc/1XKdF8fZ/Screenshot-2020-11-09-213940.png)](https://postimg.cc/sGvKdDYn)

