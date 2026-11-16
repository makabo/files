#Import Libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# %%
#Prepare the Text Data
# Sample text (e.g., Shakespeare)
text = """Shall I compare thee to a summer's day? Thou art more lovely and more
temperate.
Rough winds do shake the darling buds of May, And summer's lease hath all too
short a date."""

# %%
#Tokenize the Text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# %%
#Generate Input Sequences
# Create sequences for text generation
input_sequences = []
for line in text.split('.'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# %%
#Pad Sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences,maxlen=max_sequence_len, padding='pre'))

# %%
#Split Input and Output
# Split sequences into input (X) and output (y)
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# %%
#Build the LSTM Model
# Build LSTM model for text generation
model_textgen_lstm = Sequential()
model_textgen_lstm.add(Embedding(total_words, 64,input_length=max_sequence_len-1))
model_textgen_lstm.add(LSTM(100))
model_textgen_lstm.add(Dense(total_words, activation='softmax'))

# %%
#Compile and Train the Model
# Compile model
model_textgen_lstm.compile(optimizer='adam', loss='categorical_crossentropy',
metrics=['accuracy'])

# %%
# Train the model
model_textgen_lstm.fit(X, y, epochs=100, verbose=1)

# %%
#Generate New Text
# Generate new text
def generate_text(seed_text, next_words, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1,padding='pre')
        predicted = np.argmax(model_textgen_lstm.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# %%
#Generating Text
# Example text generation
print(generate_text("Shall I compare thee", 5, max_sequence_len))



