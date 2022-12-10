import pandas as pd

# Responsavel por transformar o dataset original
# Faz a leitura do Dataset original e divide em tres datasets diferentes (um para o server e um para cada client)
def fileTransform():
    columns = ["sentiment", "id", "date", "query", "user_id", "text"]
    df_original = pd.read_csv('./dataset/original/original.csv', encoding='latin', names=columns)

    df_original.drop(['id', 'date', 'query', 'user_id'], axis=1, inplace=True)

    # trocando valores de sentimento de numero para texto
    df_original['sentiment'] = df_original['sentiment'].map({0:'Negative', 2:'Neutral', 4:'Positive'})

    print(f'Valor de Sentimentos: {df_original.sentiment.unique()}')
    print(f'Valor NULL:\n{df_original.isna().sum()}')

    # Embaralha os dados e reseta o index
    df_original = df_original.sample(frac=1).reset_index(drop=True)

    # Pega os ultimos 5k de dados para o server
    df_server = df_original[-5000:]

    # Pega os ultimos 10k de dados para os client
    df_client1 = df_original[0:50000]
    df_client2 = df_original[100000:150000]

    df_server.to_csv('./dataset/custom/server.csv', encoding='latin', index=False)
    df_client1.to_csv('./dataset/custom/client1.csv', encoding='latin', index=False)
    df_client2.to_csv('./dataset/custom/client2.csv', encoding='latin', index=False)

    print(f"Server shape: {df_server.shape}")
    print(f"Client1 shape: {df_client1.shape}")
    print(f"Client2 shape: {df_client2.shape}")



if __name__ == '__main__':
    fileTransform()