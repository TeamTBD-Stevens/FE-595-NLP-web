from flask import Flask, render_template, request, url_for
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import string
from translate import Translator
import enchant
from langdetect import detect
# from chatterbot import ChatBot
# from chatterbot.trainers import ChatterBotCorpusTrainer
# import spacy


# spacy.load('en')
app = Flask(__name__)
nltk.download('averaged_perceptron_tagger')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/result', methods=['POST'])
def user_rec():
    checklist = request.form.getlist('s_option')
    sentiment, word_bag, language, highest_counts, tags, french, chinese, out_message\
        = ['Empty block 19491001 zhrmghgcl'] * 8
    # ai_response = 'Empty block 19491001 zhrmghgcl'

    if request.method == 'POST':
        text = request.form['message']
        if text is None or len(text.split()) == 0:
            return render_template('blank error.html')

        if len(checklist) == 0:
            return render_template('no choice error.html')

        if 'sentiment_analysis' in checklist:
            sentiment = sentiment_analysis()

        if 'language_detect' in checklist:
            language = language_detect()

        if 'useful_words' in checklist:
            word_bag = useful_words()

        if 'top_10' in checklist:
            highest_counts = top_10()

        if 'pos_tag' in checklist:
            tags = pos_tag()

        if 'translate_French' in checklist:
            french = translate_French()

        if 'translate_Chinese' in checklist:
            chinese = translate_Chinese()

        if 'spell_check' in checklist:
            out_message = spell_check()

        # if 'AI_response' in checklist:
        #     response = ai_response()

    return render_template('result.html', prediction=sentiment, words=word_bag,
                           top_10=highest_counts, tags=tags, french=french, chinese=chinese,
                           check_message=out_message, language=language
                           # , ai_response=response
                           )


def sentiment_analysis():
    message = request.form['message']
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(message)
    result = vs["compound"]
    return result


def language_detect():
    language_code = pd.read_csv('./material/language_code.csv')
    text = request.form['message']
    code = detect(text)
    if code in language_code['code'].values:
        language = language_code[language_code['code'] == code].values[0][1]
    else:
        language = code
    return language


def useful_words():
    nltk.download('stopwords')
    nltk.download('punkt')
    text = request.form['message']
    text_tokens = word_tokenize(text)
    word_bag = [word for word in text_tokens if word not in stopwords.words()]
    return word_bag


def top_10():
    sample = request.form['message']
    voc = sample.split()
    counts = pd.value_counts(voc).sort_values(ascending=False)
    highest_counts = [(counts.keys()[i], counts.values[i]) for i in range(len(counts))]
    return highest_counts


def pos_tag():
    sample = request.form['message']
    tokens = [i.strip(string.punctuation) for i in sample.split(" ")]
    tags = nltk.pos_tag(tokens)
    return tags


def translate_French():
    text = request.form['message']
    translator = Translator(to_lang="french")
    try:
        french = translator.translate(text)
    except:
        french = 'fail to translate'
    if text == french:
        french = 'fail to translate'
    return french


def translate_Chinese():
    text = request.form['message']
    translator = Translator(to_lang="chinese")
    try:
        chinese = translator.translate(text)
    except:
        chinese = 'fail to translate'
    if text == chinese:
        chinese = 'fail to translate'
    return chinese


def spell_check():
    d = enchant.Dict("en_US")
    text = request.form['message']
    tokens = [i.strip(string.punctuation) for i in text.split(' ')]
    judge_list = [d.check(word) for word in tokens]
    if all(judge_list) is True:
        out_message = 'You spell all words correct!'
    else:
        wrong_index = [i for i in range(len(judge_list)) if judge_list[i] is False]
        wrong_words = [tokens[i] for i in wrong_index]
        out_message = "Check word(s) '{0}' if this is English text.".format("', '".join(wrong_words))
    return out_message


# def ai_response():
#     text = request.form['message']
#     BankBot = ChatBot(name='BankBot',
#                       read_only=False,
#                       logic_adapters=["chatterbot.logic.BestMatch"],
#                       storage_adapter="chatterbot.storage.SQLStorageAdapter")
#     corpus_trainer = ChatterBotCorpusTrainer(BankBot)
#     corpus_trainer.train("emotion")
#     response = str(BankBot.get_response(text))
#     return response


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)

