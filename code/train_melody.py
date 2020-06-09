#!/usr/bin/python3
import random  # отсюда и до конца - блок похожий на тот, что закомментирован выше. Но здесь берется последовательность из 3 нот из рандомного места и следующая нота прелсказывается по 3 нотам со смещением на ту ноту, которую система предсказала сама
from keras.layers import LSTM
import pickle
import gensim.corpora as corpora
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
from keras.layers import Dense
from keras.models import Sequential
import numpy
from mido import MidiFile
from mido import tick2second
from os import listdir
from os.path import isfile, join
files = [f for f in listdir("./blues/") if isfile(join("./blues/", f))]
num_pred_notes = 150


# перевод в формат NOV - где N - нота; O - октава; V - длительность. Заглавная N - с диезом, строчная n - без
def convert_to_note(note, time):
    alphabet = ['c', 'C', 'd', 'D', 'e', 'f', 'F',
                'g', 'G', 'a', 'A', 'b', 'p']  # p - пауза
    octave = int(abs(note/12))
    ton = alphabet[note % 12]
    value = 0
    if (note < 0):
        ton = alphabet[12]
    if (time > 0 and time <= 0.09375):
        value = 0  # 1/16
    if (time > 0.09375 and time <= 0.1875):
        value = 1  # 1/8
    if (time > 0.1875 and time <= 0.3125):
        value = 2  # 1/4
    if (time > 0.3125 and time <= 0.4375):
        value = 3  # 3/8
    if (time > 0.4375 and time <= 0.625):
        value = 4  # 1/2
    if (time > 0.625 and time <= 0.875):
        value = 5  # 3/4
    if (time > 0.875 and time <= 1):
        value = 6  # 1
    result = str(ton) + str(octave) + str(value)
    return result


def convert_from_note(string):
    time = 0
    note = 0
    note_number = 0
    alphabet = ['c', 'C', 'd', 'D', 'e', 'f',
                'F', 'g', 'G', 'a', 'A', 'b', 'p']
    for i in range(0, len(alphabet)):
        if (string[0] == alphabet[i]):
            note = i

    if (note != 12):
        note_number = note+(int(string[1])*12)
    else:
        note_number = -1

    if (int(string[2]) == 0):
        time = 0.0625
    if (int(string[2]) == 1):
        time = 0.125
    if (int(string[2]) == 2):
        time = 0.25
    if (int(string[2]) == 3):
        time = 0.325
    if (int(string[2]) == 4):
        time = 0.5
    if (int(string[2]) == 5):
        time = 0.75
    if (int(string[2]) == 6):
        time = 1

    return(note_number, time)


seq = []  # последовательность нот, октав и их длительностей
for file in files:
    mid = MidiFile("./blues/" + file)
    ticks = mid.ticks_per_beat
    sound = 0
    note = 0
    pause = 0
    off = 0
    # цикл для получения последовательности из сообщений
    for i, track in enumerate(mid.tracks):
        for msg in track:
            if (msg.type == 'note_on'):
                if (sound == 0):
                    if (msg.time/(ticks*4) > 0.05):
                        res = convert_to_note(note, off/(ticks*4))
                        seq.append(res)
                        #print (str(note) + ' ' + str(off/(ticks*4)) +  ' ' + str(res))
                        res = convert_to_note(-1, (msg.time)/(ticks*4))
                        seq.append(res)
                        #print ('pause ' + str((msg.time)/(ticks*4)) + ' ' + str(res))
                        off = 0
                    else:
                        res = convert_to_note(note, (msg.time+off)/(ticks*4))
                        seq.append(res)
                        #print (str(note) + ' ' + str((msg.time+off)/(ticks*4)) + ' ' + str(res))

                else:
                    if (sound == 1):
                        res = convert_to_note(note, (msg.time)/(ticks*4))
                        seq.append(res)
                        #print (str(note) + ' ' + str((msg.time)/(ticks*4)) + ' ' + str(res))
                sound = sound+1
                note = msg.note
            else:
                if (msg.type == 'note_off'):
                    if (sound == 1):
                        off = msg.time
                    sound = sound-1
        res = convert_to_note(note, (off)/(ticks*4))
        seq.append(res)
        #print (str(note) + ' ' + str(off/(ticks*4)) + ' ' + str(res))
        # print(seq)

numpy.random.seed(50)
alphabet = seq


# формирование словарей из нот в номер и из номера в ноту
seq2 = numpy.reshape(seq, (len(seq), 1))
int_to_note = corpora.Dictionary(seq2)
note_to_int = {}
for key, value in int_to_note.items():
    note_to_int.setdefault(value, key)

seq_length = 10  # это из лекции
dataX = []
dataY = []
for i in range(0, len(alphabet) - seq_length, 1):
    seq_in = alphabet[i:i + seq_length]
    seq_out = alphabet[i + seq_length]
    dataX.append([note_to_int[note] for note in seq_in])
    dataY.append(note_to_int[seq_out])
    print(seq_in, '->', seq_out)


X = numpy.reshape(dataX, (len(dataX), 1, seq_length))  # и это из лекции
X = X / float(len(alphabet))
print(X)
y = np_utils.to_categorical(dataY)
model = Sequential()
model.add(LSTM(128, input_shape=(
    X.shape[1], X.shape[2]), return_sequences=True))
model.add(LSTM(128, input_shape=(
    X.shape[1], X.shape[2]), return_sequences=False))
model.add(LeakyReLU(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=100, batch_size=1, verbose=2)
scores = model.evaluate(X, y, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))

r = int(random.uniform(0, len(dataX)/2))
data_train = numpy.reshape(dataX[r], (1, 1, len(dataX[r])))
generated_melody = []
for i in range(0, seq_length-1):
    generated_melody.append(int_to_note[data_train[0][0][i]])
data_train = data_train/(float(len(alphabet)))

for i in range(0, num_pred_notes):
    prediction = model.predict(data_train, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_note[index]
    generated_melody.append(result)
    data_train[0][0][0] = data_train[0][0][1]
    data_train[0][0][1] = data_train[0][0][2]
    data_train[0][0][2] = data_train[0][0][3]
    data_train[0][0][3] = data_train[0][0][4]
    data_train[0][0][4] = data_train[0][0][5]
    data_train[0][0][5] = data_train[0][0][6]
    data_train[0][0][6] = data_train[0][0][7]
    data_train[0][0][7] = data_train[0][0][8]
    data_train[0][0][8] = data_train[0][0][9]
    data_train[0][0][9] = index/(float(len(alphabet)))
    print(data_train)


print(generated_melody)

save = pickle.dumps(model)
f = open("model", "wb")
f.write(save)
f.close()
