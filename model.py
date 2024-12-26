import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Dense, LSTM, GRU, Dropout, SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
import joblib

# Load dataset
data = pd.read_csv('processed_reviews.csv')

# Preprocessing: Fill missing values and adjust ratings
data['processed_review'] = data['processed_review'].fillna('')
data['rating'] = data['rating'].apply(lambda x: int(x.split()[0]) - 1 if isinstance(x, str) else x)

# Ensure all rating categories are included
if len(data['rating'].unique()) < 5:
    for i in range(5):
        if i not in data['rating'].unique():
            data = pd.concat([data, pd.DataFrame({'processed_review': ['placeholder'], 'rating': [i]})], ignore_index=True)

# Input and labels
X = data['processed_review']
y = data['rating']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Tokenization and padding
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding
max_len = 100
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

# Save tokenizer
joblib.dump(tokenizer, 'tokenizer.joblib')

# Number of classes
num_classes = len(set(y))

# LSTM Model
def create_lstm_model():
    model = Sequential([
        Embedding(input_dim=10000, output_dim=128, input_length=max_len),
        SpatialDropout1D(0.2),
        LSTM(128, dropout=0.3, recurrent_dropout=0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

lstm_model = create_lstm_model()
history_lstm = lstm_model.fit(X_train_pad, y_train, 
                              validation_data=(X_test_pad, y_test), 
                              epochs=10, batch_size=64, verbose=1, 
                              callbacks=[
                                  tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])

# Save LSTM model
lstm_model.save('lstm_model.h5')

# Evaluate LSTM
y_pred_lstm = lstm_model.predict(X_test_pad).argmax(axis=1)
print("LSTM Accuracy:", accuracy_score(y_test, y_pred_lstm))
print(classification_report(y_test, y_pred_lstm))

# Plot LSTM training results
def plot_lstm_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history.history['loss'], label='Training Loss', linestyle='--')
    plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--')
    plt.title("LSTM Training History")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy / Loss")
    plt.legend()
    plt.show()

plot_lstm_history(history_lstm)

# GRU Model
def create_gru_model():
    model = Sequential([
        Embedding(input_dim=10000, output_dim=128, input_length=max_len),
        SpatialDropout1D(0.2),
        GRU(128, dropout=0.3, recurrent_dropout=0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

gru_model = create_gru_model()
history_gru = gru_model.fit(X_train_pad, y_train, 
                            validation_data=(X_test_pad, y_test), 
                            epochs=10, batch_size=64, verbose=1, 
                            callbacks=[
                                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])

# Save GRU model
gru_model.save('gru_model.h5')

# Evaluate GRU
y_pred_gru = gru_model.predict(X_test_pad).argmax(axis=1)
print("GRU Accuracy:", accuracy_score(y_test, y_pred_gru))
print(classification_report(y_test, y_pred_gru))

# Plot GRU training results
def plot_gru_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history.history['loss'], label='Training Loss', linestyle='--')
    plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--')
    plt.title("GRU Training History")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy / Loss")
    plt.legend()
    plt.show()

plot_gru_history(history_gru)

# Transformer Model
def create_transformer_model():
    inputs = tf.keras.Input(shape=(max_len,))
    x = Embedding(input_dim=10000, output_dim=128)(inputs)
    attention = MultiHeadAttention(num_heads=4, key_dim=128)(x, x)
    x = LayerNormalization(epsilon=1e-6)(attention + x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

transformer_model = create_transformer_model()
history_transformer = transformer_model.fit(X_train_pad, y_train, 
                                            validation_data=(X_test_pad, y_test), 
                                            epochs=10, batch_size=64, verbose=1, 
                                            callbacks=[
                                                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])

# Save Transformer model
transformer_model.save('transformer_model.h5')

# Evaluate Transformer
y_pred_transformer = transformer_model.predict(X_test_pad).argmax(axis=1)
print("Transformer Accuracy:", accuracy_score(y_test, y_pred_transformer))
print(classification_report(y_test, y_pred_transformer))

# Plot Transformer training results
def plot_transformer_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history.history['loss'], label='Training Loss', linestyle='--')
    plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--')
    plt.title("Transformer Training History")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy / Loss")
    plt.legend()
    plt.show()

plot_transformer_history(history_transformer)
