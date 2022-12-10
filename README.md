# NPL-com-LSTM-e-FL

Projeto de final da materia de INF317E - Tópicos em Inteligência Artificial: Aprendizagem Profunda 2022.3 - UFABC

## Descrição

Inicio dos estudos na area de Federated Leraning.
O projeto tem como objetivo a sumulação de um ambiente de aprendizado federado com dados de tweets, o ambiente é divido em 3 agentes (1 servidor e 2 clientes) com dados distintos em cada um deles.

## Getting Started

### Dependencias

* python > 3.6


### Installing

* Instalação das dependencias
* Descompactação do dataset
```
pip install -r requirements.txt
unzip -o src/dataset.zip
```

### Executing program

* Re-execução do dataset (opcional)
```
python3 src/utils.py
```

Será necessario abrir 3 terminais diferentes para execução do codigo (um para cada agente)
* Inicialização do servidor
```
python3 server.py
```
* Inicialização do cliente1
```
python3 client1.py
```
* Inicialização do cliente2
```
python3 client1.py
```
