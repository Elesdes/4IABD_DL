import os
from datetime import date

def create_tensorboard_log_dir(type_of_algorithm: str = None,
                           algorithm_used: str = None,
                           batch_size: int = None,
                           learning_rate: int = None,
                           momentum: int = None,
                           num_hidden_layers: int = None,
                           num_neurons_per_hidden_layers: int = None) -> str:
    base_dir_logs = "../../tensorboard_logs"
    experiment_dir_logs = f"{date.today().strftime('%d%m%Y')}" \
                          f"_bs_{batch_size}" \
                          f"_lr_{learning_rate}" \
                          f"_mom_{momentum}" \
                          f"_hl_{num_hidden_layers}" \
                          f"_nphl_{num_neurons_per_hidden_layers}"
    paths = [f"{base_dir_logs}",
             f"{base_dir_logs}/{type_of_algorithm}",
             f"{base_dir_logs}/{type_of_algorithm}/{algorithm_used}",
             f"{base_dir_logs}/{type_of_algorithm}/{algorithm_used}/{experiment_dir_logs}"]

    for path in paths:
        if not os.path.isdir(path):
            os.mkdir(path)
            print("created", path)

    return paths[3]

if __name__ == "__main__":
    create_tensorboard_log_dir("ML", "learning")