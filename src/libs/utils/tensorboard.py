import os
from datetime import date

from src.libs.utils.default import default_parameters

from random_word import RandomWords


@default_parameters
def create_tensorboard_log_dir(type_of_algorithm: str = "Test",
                               algorithm_used: str = "Test",
                               batch_size: int = default_parameters.batch_size,
                               learning_rate: float = None,
                               momentum: float = None,
                               num_hidden_layers: int = None,
                               num_neurons_per_hidden_layers: int = None):
    """type_of_algorithm: DL or ML - Default: Test
    || algorithm_used: CNN, RNN or transformers for DL or LINEAR or MLP for ML - Default: Test
    || every parameters are optionals, but if set to None, then a random generated word is put in place"""

    def decorator(function):
        def wrapper(path: str = None, *args):
            # Relative path from this file to tensorboard log directory
            base_dir_logs = "../../../tensorboard_logs"

            # Create a path file name with date and other variables
            experiment_dir_logs = f"{date.today().strftime('%d%m%Y')}" \
                                  f"_bs_{batch_size}" \
                                  f"_lr_{learning_rate if learning_rate is not None else RandomWords().get_random_word()}" \
                                  f"_mom_{momentum if momentum is not None else RandomWords().get_random_word()}" \
                                  f"_hl_{num_hidden_layers if num_hidden_layers is not None else RandomWords().get_random_word()}" \
                                  f"_nphl_{num_neurons_per_hidden_layers if num_neurons_per_hidden_layers is not None else RandomWords().get_random_word()}"

            paths = [f"{base_dir_logs}",
                     f"{base_dir_logs}/{type_of_algorithm}",
                     f"{base_dir_logs}/{type_of_algorithm}/{algorithm_used}",
                     f"{base_dir_logs}/{type_of_algorithm}/{algorithm_used}/{experiment_dir_logs}"]

            # Creating every dirs that doesn't exist yet
            for path in paths:
                if not os.path.isdir(path):
                    os.mkdir(path)

            return function(path=paths[3], *args)

        return wrapper

    return decorator


@create_tensorboard_log_dir(type_of_algorithm="how_to_use")
def how_to_use(path: str = None) -> None:
    print(f"Path created at : {path}")


if __name__ == "__main__":
    how_to_use()
