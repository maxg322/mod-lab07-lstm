import numpy as np
import random
import sys

from tensorflow.keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from tensorflow.keras.optimizers import RMSprop

# Загрузка текста
with open('src/input.txt', 'r', encoding='utf-8') as file:
    text = file.read().lower()

# Токенизация по словам
textWords = text.split()
vocabulary = sorted(set(textWords))

# Словари: слово <-> индекс
word_to_indices = {w: i for i, w in enumerate(vocabulary)}
indices_to_word = {i: w for i, w in enumerate(vocabulary)}

# Подготовка данных
max_length = 10
sequences = []
next_words = []
for i in range(len(textWords) - max_length):
    sequences.append(textWords[i:i + max_length])
    next_words.append(textWords[i + max_length])

# Векторизация
X = np.zeros((len(sequences), max_length, len(vocabulary)), dtype=bool)
y = np.zeros((len(sequences), len(vocabulary)), dtype=bool)
for i, seq in enumerate(sequences):
    for t, w in enumerate(seq):
        X[i, t, word_to_indices[w]] = 1
    y[i, word_to_indices[next_words[i]]] = 1

# Построение модели
model = Sequential()
model.add(LSTM(128, input_shape=(max_length, len(vocabulary))))
model.add(Dense(len(vocabulary)))
model.add(Activation('softmax'))
optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# Температурная функция
def sample_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    log_preds = np.log(np.maximum(preds, 1e-8)) / temperature
    exp_preds = np.exp(log_preds - np.max(log_preds))
    probs = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(probs), p=probs)

# Обучение модели
model.fit(X, y, batch_size=128, epochs=50)

# Генерация текста
def generate_text(length, diversity=1.0):
    start_idx = random.randint(0, len(textWords) - max_length - 1)
    seed = textWords[start_idx:start_idx + max_length]
    output = seed.copy()
    for _ in range(length):
        x_pred = np.zeros((1, max_length, len(vocabulary)))
        for t, w in enumerate(seed):
            x_pred[0, t, word_to_indices[w]] = 1
        preds = model.predict(x_pred, verbose=0)[0]
        idx = sample_temperature(preds, diversity)
        w = indices_to_word[idx]
        output.append(w)
        seed = seed[1:] + [w]
    return ' '.join(output)

# Генерация и сохранение
result_text = generate_text(1000, diversity=0.7)
with open('result/gen.txt', 'w', encoding='utf-8') as f:
    f.write(result_text)
print("Генерация завершена. Результат сохранён в result/gen.txt")
