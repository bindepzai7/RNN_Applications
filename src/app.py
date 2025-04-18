import streamlit as st
import torch
import torch.nn as nn
import pickle
import re
import unidecode
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from torchtext.data.utils import get_tokenizer

from SentimentClassifier import SentimentClassifier
from WeatherForecastor import WeatherForecastModel

st.set_page_config(page_title="RNN Apllications", page_icon="🌍")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_vocab():
    with open('./models/vocab.pkl', 'rb') as f:
        vocab_list = pickle.load(f)
    return {word: idx for idx, word in enumerate(vocab_list)}

@st.cache_resource
def load_model1(vocab_size):
    model = SentimentClassifier(vocab_size=vocab_size, embedding_dim=64, hidden_size=64, n_layers=2, n_classes=3, dropout_prob=0.2)
    model.load_state_dict(torch.load('./models/problem1.pth', map_location=device))
    model.eval()
    return model

@st.cache_resource
def load_model2():
    model = WeatherForecastModel(embedding_dim=1, hidden_size=8, n_layers=3, dropout_prob=0.2)
    model.load_state_dict(torch.load('./models/problem2.pth', map_location=device))
    model.eval()
    return model

vocab = load_vocab()
model1 = load_model1(len(vocab))
model2 = load_model2()

tokenizer = get_tokenizer('basic_english')
english_stop_words = stopwords.words('english')
stemmer = PorterStemmer()

def text_normalize(text):
    text = text.lower()
    text = unidecode.unidecode(text)
    text = text.strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in english_stop_words])
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

def plot_temperature_sequence(inputs, prediction):
    plt.figure(figsize=(8, 4))
    time = list(range(len(inputs)))
    plt.plot(time, inputs, marker='o', label='Past Temperatures')
    plt.plot(len(inputs), prediction, marker='X', color='red', label='Predicted Next Hour')
    plt.title("Temperature Forecast")
    plt.xlabel("Time Step (Hour)")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    st.pyplot(plt)

def plot_sentiment_confidence(probabilities, sentiment_map):
    st.bar_chart({sent: float(prob) for sent, prob in zip(sentiment_map.values(), probabilities)})

def run_sentiment_analysis():
    st.header("📊 Financial News Sentiment Analysis")
    st.markdown("**Model loaded with vocab size:**")
    st.write(len(vocab))
    st.markdown("**Special tokens available:**")
    st.write([word for word in vocab if word.upper() in ['PAD', 'UNK']])

    text_input = st.text_area("Enter your financial news text here:", height=200)

    if st.button('Analyze Sentiment'):
        if text_input:
            normalized_text = text_normalize(text_input)
            tokens = tokenizer(normalized_text)

            max_seq_len = 25
            indices = []
            unknown_tokens = []
            for token in tokens:
                if token in vocab:
                    indices.append(vocab[token])
                else:
                    indices.append(vocab.get('UNK', 0))
                    unknown_tokens.append(token)

            if unknown_tokens:
                st.warning(f"{len(unknown_tokens)} unknown token(s): {unknown_tokens}")

            indices = indices[:max_seq_len] + [vocab.get('PAD', 0)] * max(0, max_seq_len - len(indices))
            tensor = torch.LongTensor(indices).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model1(tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                prediction = torch.argmax(output, dim=1).item()

            sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
            sentiment = sentiment_map[prediction]

            st.success(f"Predicted Sentiment: {sentiment}")
            st.markdown("### Confidence Scores:")
            plot_sentiment_confidence(probabilities, sentiment_map)
        else:
            st.warning("Please enter some text to analyze.")

def run_weather_forecasting():
    st.header('🌤 Hourly Temperature Forecasting')

    example = [24.86, 22.75, 20.07, 17.81, 17.16, 15.01]
    target = 14.47
    st.markdown("### Example Input")
    st.write(f"Input Temperatures: {example}")
    st.write(f"Expected Target Temperature: {target}°C")

    user_input = st.text_input("Enter 6 temperatures separated by commas:", ", ".join(map(str, example)))

    if st.button('Predict Temperature'):
        try:
            temp_data = [float(temp.strip()) for temp in user_input.split(",")]
            if len(temp_data) != 6:
                st.error("Please enter exactly 6 temperature values.")
                return

            temp_tensor = torch.FloatTensor(temp_data).unsqueeze(0).unsqueeze(-1).to(device)
            with torch.no_grad():
                prediction = model2(temp_tensor)
                predicted_temp = prediction.item()

            st.success(f"Predicted Temperature for the Next Hour: {predicted_temp:.2f}°C")
            plot_temperature_sequence(temp_data, predicted_temp)
        except ValueError:
            st.error("Invalid input. Please ensure all values are numbers separated by commas.")

def main():
    st.title("🌐 Various RNN Applications")
    tab1, tab2, tab3 = st.tabs(["📊 Sentiment Analysis", "🌤 Weather Forecasting", "Other Tasks"])

    with tab1:
        run_sentiment_analysis()
    with tab2:
        run_weather_forecasting()
    with tab3:
        st.header("Other Tasks")
        st.markdown("Comming soon...")

if __name__ == "__main__":
    main()
    st.markdown("Made with ❤️ by bindepzai")
