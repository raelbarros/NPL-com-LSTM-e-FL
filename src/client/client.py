import flwr as fl


# Classe modelo de client
class Client(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    
    def fit(self, parameters, config):  
        '''
            Função de Treino
            parameters: parametros do modelo (compartilhado com o global)
            config: configurações de treino (compartilhado com o global)
        '''
        print("\n\n\n----------------  Train ----------------- ")

        self.model.set_weights(parameters)

        epochs = config["local_epochs"]

        history = self.model.fit(self.x_train, 
            self.y_train, 
            epochs=epochs,
            validation_split=0.2
        )
        
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }

        return self.model.get_weights(), len(self.x_train), results

    def evaluate(self, parameters, config):
        '''
            Função de Teste
            parameters: parametros do modelo (compartilhado com o global)
            config: configurações de teste (compartilhado com o global)
        '''
        print("\n\n\n----------------  Test ----------------- ")

        self.model.set_weights(parameters)

        steps = config["val_steps"]

        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, steps=steps)
        return loss, len(self.x_test), {"accuracy": accuracy}
