#
class default_parameters:
    "Get default values for max_size, num_words, chunk_size, iterations, embedding_dim, epochs or batch_size. Can be modified."
    max_size = 100
    num_words = 10000
    chunk_size = 100000
    iterations = 10
    embedding_dim = 16
    epochs = 250
    batch_size = 512

    def __init__(self, function):
        self.function = function

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


@default_parameters
def how_to_use() -> None:
    print(default_parameters.max_size)
    default_parameters.max_size = 12
    print(default_parameters.max_size)


if __name__ == "__main__":
    how_to_use()
