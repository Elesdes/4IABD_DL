#
class DefaultParameters:
    """Get default values for learning_rate, momentum, num_hidden_layers, num_neurons_per_hidden_layers,
    iterations, epochs, batch_size
    max_size, num_words, chunk_size, embedding_dim.
    Can be modified."""
    # These variables are about models used
    learning_rate = 0.01
    momentum = 0.01
    num_hidden_layers = 1
    num_neurons_per_hidden_layers = 1

    # These variables are for training and testing purposes
    iterations = 10
    epochs = 250
    batch_size = 512

    # These values are about data manipulation
    max_size = 100
    num_words = 10000
    chunk_size = 100000
    embedding_dim = 16


def how_to_use() -> None:
    print(DefaultParameters.max_size)
    DefaultParameters.max_size = 12
    print(DefaultParameters.max_size)


if __name__ == "__main__":
    how_to_use()
