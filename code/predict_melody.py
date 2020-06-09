#!/usr/bin/python3
import random  # здесь берется последовательность из 10 нот из рандомного места и следующая нота предсказывается по 10 нотам со смещением на ту ноту, которую система предсказала сама
from mido import MidiFile
from mido import Message, MidiFile, MidiTrack
import gensim.corpora as corpora
from keras.utils import np_utils
import numpy
from os.path import isfile, join
from os import listdir
from mido import tick2second
import sys
from sklearn.datasets import load_files
import pickle

f = open("model", "rb")
s = f.read()
f.close()
model = pickle.loads(s)

files = [f for f in listdir("./blues/") if isfile(join("./blues/", f))]
num_pred_notes = 150


# перевод в формат NOV - где N - нота; O - октава; V - длительность. Заглавная N - с диезом, строчная n - без
def convert_to_note(note, time):
    if (time == 0):
        return ''
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
        if(string == ''):
            return (-1, 0)
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

seq_length = 10
dataX = []
dataY = []
for i in range(0, len(alphabet) - seq_length, 1):
    seq_in = alphabet[i:i + seq_length]
    seq_out = alphabet[i + seq_length]
    dataX.append([note_to_int[note] for note in seq_in])
    dataY.append(note_to_int[seq_out])
    print(seq_in, '->', seq_out)

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


print(generated_melody)


mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)
mid.ticks_per_beat = 192

t = 10
track.append(Message('program_change', program=12, time=0))
for i in range(0, len(generated_melody)):
    gen_note, gen_time = convert_from_note(generated_melody[i])
    if (gen_note > 0):
        track.append(Message('note_on', note=gen_note,
                             velocity=100, time=int(t)))
        track.append(Message('note_off', note=gen_note,
                             velocity=0, time=int(gen_time*768)))
        t = 10
    else:
        t = gen_time*768

mid.save('pred_melody5.mid')
