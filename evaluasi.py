import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import joblib
import pickle

# Load preprocessed data
print("Loading data...")
data = pd.read_csv('processed_reviews.csv')

# Handle missing values
data['processed_review'] = data['processed_review'].fillna('')

# Prepare data
print("Preparing data...")
X = data['processed_review']
y = data['rating'].apply(lambda x: 1 if int(x.split()[0]) <= 2 else 0)  # Binary classification: 1 for Negative, 0 for Positive

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenizer and sequence preparation
print("Tokenizing data...")
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_pad = pad_sequences(X_train_seq, maxlen=100)
X_test_pad = pad_sequences(X_test_seq, maxlen=100)

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# Function to plot accuracy and loss
def plot_metrics(history, model_name):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.legend()

    plt.show()

# 1. LSTM Model
print("Training LSTM model...")
model_lstm = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=100),
    LSTM(128),
    Dense(1, activation='sigmoid')
])
model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_lstm = model_lstm.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_data=(X_test_pad, y_test), callbacks=[early_stopping])
plot_metrics(history_lstm, "LSTM")

# Evaluate LSTM
print("Evaluating LSTM model...")
y_pred_lstm = (model_lstm.predict(X_test_pad) > 0.5).astype('int32')
print("LSTM Classification Report")
print(classification_report(y_test, y_pred_lstm))

# Save LSTM model
model_lstm.save('lstm_model.h5')
print("LSTM model saved as lstm_model.h5")

# 2. GRU Model
print("Training GRU model...")
model_gru = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=100),
    GRU(128),
    Dense(1, activation='sigmoid')
])
model_gru.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_gru = model_gru.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_data=(X_test_pad, y_test), callbacks=[early_stopping])
plot_metrics(history_gru, "GRU")

# Evaluate GRU
print("Evaluating GRU model...")
y_pred_gru = (model_gru.predict(X_test_pad) > 0.5).astype('int32')
print("GRU Classification Report")
print(classification_report(y_test, y_pred_gru))

# Save GRU model
model_gru.save('gru_model.h5')
print("GRU model saved as gru_model.h5")

# Save tokenizer using joblib
joblib.dump(tokenizer, 'tokenizer.pkl')
print("Tokenizer saved as tokenizer.pkl")

# Compare Accuracy
accuracy_lstm = accuracy_score(y_test, y_pred_lstm)
accuracy_gru = accuracy_score(y_test, y_pred_gru)

print(f"LSTM Accuracy: {accuracy_lstm}")
print(f"GRU Accuracy: {accuracy_gru}")

# Select the best model
if accuracy_lstm > accuracy_gru:
    print("LSTM is the better model.")
else:
    print("GRU is the better model.")

# Optionally, you can also save the models and tokenizer using pickle (if needed)
with open('lstm_model.pkl', 'wb') as f:
    pickle.dump(model_lstm, f)
print("LSTM model saved using pickle as lstm_model.pkl")

with open('gru_model.pkl', 'wb') as f:
    pickle.dump(model_gru, f)
print("GRU model saved using pickle as gru_model.pkl")
