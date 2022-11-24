import flwr as fl
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from src.client.client import Client
from src.model.model import Model

def main():
    # Carrega dataset
    df = pd.read_csv('./src/dataset/custom/client1.csv', encoding='latin')

    # Pega valores de x, y
    x = df['text'].astype(str)
    y = df['sentiment'].values

    # Split dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # Tokeniza texto
    token = Tokenizer(num_words=1920, oov_token='x')
    token.fit_on_texts(x_train)
    token.fit_on_texts(x_test)

    # Converte palavras em chave
    x_train = pad_sequences(token.texts_to_sequences(x_train))
    x_test = pad_sequences(token.texts_to_sequences(x_test))

    # Enconder
    encoder = LabelEncoder()
    encoder.fit(y)
    y_train = encoder.transform(y_train)
    y_test = encoder.transform(y_test)

    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)

    # Carrega modelo
    model = Model().create_model()

    # Instancia o Client
    client = Client(model, x_train, y_train, x_test, y_test)

    # Inicia Flower Client
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=client
    )


if __name__ == "__main__":
    main()
