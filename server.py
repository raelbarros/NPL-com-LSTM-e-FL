import flwr as fl
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.preprocessing import LabelEncoder
from src.server.server import Server
from src.model.model import Model

def main():
    # Carrega modelo
    model = Model().create_model()

    # Criacao de estrategia do servidor
    strategy = Server(
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
    )

    # Inicia Flower Server com 4 rounds
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=4),
        strategy=strategy
    )


def get_evaluate_fn(model):
    # Funcao para teste executada no proprio servidor

    # Carrega dataset
    df = pd.read_csv('./src/dataset/custom/server.csv', encoding='latin')

    # Pega valores de x, y
    x = df['text'].astype(str)
    y = df['sentiment'].values

    # Tokeniza texto
    token = Tokenizer(num_words=1920, oov_token='x')
    token.fit_on_texts(x)

    # Converte palavras em chave
    x = pad_sequences(token.texts_to_sequences(x))

    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    y = y.reshape(-1,1)

    # Funcao `evaluate` vai ser chamada em cada round
    def evaluate(server_round, parameters, config):
        # Atualiza o modelo com os parametros dos clientes
        model.set_weights(parameters)  
        loss, accuracy = model.evaluate(x, y)
        return loss, {"accuracy": accuracy}

    return evaluate


def fit_config(server_round):
    # Retorna configuracao de treino para os client
    return {"local_epochs": 2}


def evaluate_config(server_round):
    # Retorna configuracao de teste para os client 
    return {"val_steps": 5}


if __name__ == "__main__":
    main()