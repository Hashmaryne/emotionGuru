from email.mime.text import MIMEText
import time
import numpy
import tflearn
import tensorflow
import random
import json
import pymongo
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix, classification_report
from flask import Flask, jsonify, request, session
from flask_restful import Api, Resource
import pymongo
from werkzeug.security import generate_password_hash, check_password_hash
import random
from datetime import datetime, timedelta
from flask_cors import CORS, cross_origin
import smtplib
from email.mime.multipart import MIMEMultipart
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from textblob import Blobber
import os
from twilio.rest import Client


# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure

stemmer = LancasterStemmer();

nltk.download('punkt')
nltk.download('stopwords')

client = pymongo.MongoClient(
    "mongodb+srv://Hashmaryne:1234@cluster0.gbq8u.mongodb.net/EmotionTracker?retryWrites=true&w=majority")
db = client.EmotionTracker

chatbotintents = db.chatbotintents
activities = db.Activities
userTable = db.Users
entriesTable = db.Entries

# y = db.chatbotintents.find_one_and_update({"name": "recommendations"}, {"$addToSet": { "responses": x}})

words = []
classes = []
documents = []

english_stopwords = set(stopwords.words('english'))

# tokenizer 
tokenizer = nltk.RegexpTokenizer("[\w']+")

for intent in chatbotintents.find():
    for pattern in intent['patterns']:
        tokens = tokenizer.tokenize(pattern)  # tokenize pattern
        words.extend(tokens)  # add tokens to list
        documents.append((tokens, intent['name']))
        if intent['name'] not in classes:
            classes.append(intent['name'])

# stemming
words = [stemmer.stem(w.lower()) for w in words if w not in english_stopwords]
#remove duplicates
words = sorted(list(set(words)))

# remove duplicate classes
classes = sorted(list(set(classes)))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "words", words)

data_set = []

output_empty = [0] * len(classes)

for document in documents:
    bag = []

    # stem the pattern words for each document element
    pattern_words = document[0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    data_set.append([bag, output_row])

    '''
    print("bag", bag)
    print("output_row", output_row)
    print("data_set", data_set)
    '''

random.shuffle(data_set)
data_set = numpy.array(data_set)

train_x = list(data_set[:, 0])
train_y = list(data_set[:, 1])


tensorflow.compat.v1.reset_default_graph

#neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')


# training model
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')

pickle.dump({'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y},
            open("training_data", "wb"))
_y = model.predict(train_x)
y = train_y
'''
confusion = tensorflow.math.confusion_matrix(labels=_y, predictions=y)
confusion_matrix(_y, y)
print(confusion)
'''

y_pred=numpy.argmax(y, axis=1)
y_test=numpy.argmax(_y, axis=1)
cm = confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred, target_names=classes))
print(cm)
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


# chat()


app = Flask(__name__)
app.secret_key = 'c\x82\xd4\xf1\x85[\xb0-\xe0\xd4~\xdc\xf9\x95\x1d\x89\xb6\xf1\xae#\x14\x82\x10\x8e'

api = Api(app)
cors = CORS(app)


def start_session(user):
    session['logged in'] = True
    session['user'] = user
    return "sucess"


@app.route('/user/signup', methods=['POST'])
def signup():
    username = request.json['username']
    hashed_pass = generate_password_hash(request.json['password'], method='sha256')
    fullname = request.json['fullname']
    emergency = request.json['emergency']
    password = hashed_pass
    if userTable.find_one({'username': username}):
        return "username already exists", 400
    if userTable.insert_one({'username': username, 'password': password, 'fullname':fullname,'emergency':emergency}):
        # new_user = userTable.find_one({'username': username})
        # output = {'username': new_user['username'], 'password': new_user['password']}
        return start_session(username)

    return "Signup failed"


@app.route('/user/signout')
def signout():
    session.clear()
    return jsonify({"message": "signed out"})


@app.route('/user/login', methods=['POST'])
def login():
    username = request.json['username']
    password = request.json['password']
    auth = request.authorization
    user_found = userTable.find_one({'username': username})
    if user_found:
        user_val = user_found['username']
        passwordcheck = user_found['password']

        if (check_password_hash(passwordcheck, password)):
            return start_session(user_val)
        else:
            return "incorrect credentials"
    else:
        return "user not found"


@app.route('/emotionEntry', methods=['POST'])
def createEntry():
    activity = request.json['activity']
    emotion = request.json['emotion']
    username = request.json['username']
    people = request.json['people']
    location = request.json['location']
    today = datetime.now()
    date = today.strftime("%m/%d/%Y")
    if entriesTable.insert_one(
            {'activity': activity, 'emotion': emotion, 'username': username, 'people': people, 'location': location,
             'date': date}):
        return "entry created"
    else:
        return "entry failed"


@app.route('/addActivity', methods=['POST'])
def createActivity():
    activity = request.json['activity']
    emotion = request.json['emotion']
    username = request.json['username']
    if activities.insert_one(
            {'activity': activity, 'emotion': emotion, 'username': username}):
        return "activity created"
    else:
        return "activity failed"


@app.route('/chat', methods=['POST'])
def chat():
    responseList = []
    inputmsg = request.json['inputmsg']
    username = request.json['username']
    result = activities.find({"emotion": 'Happy', "username": username}, {'activity': 1, "_id": False})
    recommendedActivities = ([positive['activity'] for positive in result])
    results = model.predict([bag_of_words(inputmsg, words)])
    results_index = numpy.argmax(results)
    tag = classes[results_index]

    rs = userTable.find_one({ "username": username}, {'emergency': 1, 'fullname':1, "_id": False})
    emergency=rs['emergency']
    name=rs['fullname']
    msg = MIMEMultipart()
    body = "Your friend " + name + " is having suicidal thoughts, They have added you as their emergency contact. Please help them or direct them to a mental health professional"
    msg['From'] = 'sithivili.project@gmail.com'
    msg['To'] = emergency
    msg['Subject'] = 'Your friend needs help!'
    msg.attach(body)

    account_sid = 'ACb49426278280835273f316034bb19c06'
    auth_token = 'a60189ecaac78878ffa10177b533876d'
    client = Client(account_sid, auth_token)
    for tg in chatbotintents.find():
        if tg['name'] == tag:
            responses = tg['responses']
            print(tag)
    if tag=="suicide":
        '''
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login("sithivili.project@gmail.com", "skepseis123")
        server.sendmail("sithivili.project@gmail.com", emergency, msg.as_string())
        server.quit()

        '''

        client.messages \
            .create(
            body=body,
            from_='+12058274562',
            to=emergency
        )

    if tag == "sad":
        responseList.append(random.choice(responses))
        responseList.append("Here is an activity that you can do which has made you happy before,")
        responseList.append(random.choice(recommendedActivities))
        return jsonify({"message": responseList,"tag":tag})
    else:
        responseList.append(random.choice(responses))
        return jsonify({"message": responseList, "tag":tag})


@app.route('/chartData', methods=['POST'])
def chart():
    username = request.json['username']
    pastWeek = []
    positiveCount=[]
    negativeCount = []
    now = datetime.now()
    for x in range(7):
        d = now - timedelta(days=x)
        pastWeek.append(d.strftime("%m/%d/%Y"))

    for y in range(len(pastWeek)):
        result = entriesTable.find({"date": pastWeek[y], "username": username, "emotion": "Happy"}, {'emotion': 1, 'activity': 1,  "_id": False})
        #print(pastWeek[y])
        positiveCount.append(result.count())
        #print(result.count())

    for i in range(len(positiveCount)):
        print(positiveCount[i])

    for y in range(len(pastWeek)):
        result = entriesTable.find({"date": pastWeek[y], "username": username, "emotion": "Sad"}, {'emotion': 1, 'activity': 1,  "_id": False})
        #print(pastWeek[y])
        negativeCount.append(result.count())
        #print(result.count())

    for i in range(len(negativeCount)):
        print(negativeCount)

    return jsonify({"dates": pastWeek, "posCount":positiveCount, "negCount":negativeCount })

@app.route('/pieChartData', methods=['POST'])
def pieChart():
    username = request.json['username']
    pastWeek = []
    positiveCount = 0
    negativeCount = 0
    now = datetime.now()
    for x in range(7):
        d = now - timedelta(days=x)
        pastWeek.append(d.strftime("%m/%d/%Y"))

    for y in range(len(pastWeek)):
        result = entriesTable.find({"date": pastWeek[y], "username": username, "emotion": "Happy"},
                                   {'emotion': 1, 'activity': 1, "_id": False})
        # print(pastWeek[y])
        positiveCount=positiveCount+result.count()

    for y in range(len(pastWeek)):
        result = entriesTable.find({"date": pastWeek[y], "username": username, "emotion": "Sad"},
                                   {'emotion': 1, 'activity': 1, "_id": False})
        # print(pastWeek[y])
        negativeCount = negativeCount + result.count()

    return jsonify({"posCounter" : positiveCount, "negCounter": negativeCount})

@app.route('/fbPosts', methods=['POST'])
def getfbPosts():
    posts = request.json['posts']
    counter=[]
    negCounter=0;
    posCounter=0;

    data = posts['posts']
    print (data)
    tb = Blobber(analyzer=NaiveBayesAnalyzer())
    for x in range(len(data)):
        b = tb(data[x])
        y=TextBlob(data[x])
        print(y.sentiment)
        print(b.sentiment.classification)
        if(y.sentiment.polarity>=0.6):
            counter.append(1)
        else:
            counter.append(-1)

    average = sum(counter) / len(counter)
    print(average)
    avgSentiment=''
    if average<0:
        avgSentiment='neg'
    else:
        avgSentiment='pos'
    print(avgSentiment)
    return avgSentiment


@app.route('/fbChart', methods=['POST'])
def getfbChart():
    posts = request.json['posts']
    counter=[]
    negCounter=0;
    posCounter=0;

    data = posts['posts']
    print (data)
    tb = Blobber(analyzer=NaiveBayesAnalyzer())
    for x in range(len(data)):
        b = tb(data[x])
        y=TextBlob(data[x])
        print(y.sentiment.polarity)
        #print(b.sentiment.classification)
        if(y.sentiment.polarity>0):
            print("pos")
            posCounter=posCounter+1
        else:
            negCounter = negCounter + 1
            print("neg")

    print(negCounter)
    print(posCounter)
    return jsonify({"posCount":posCounter, "negCount":negCounter })

if __name__ == '__main__':
    app.run(debug=True)
