import yaml
import matplotlib.pyplot as plt

def set_params():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

def get_config(config_path='./simulation_config.yml'):
    with open(config_path, 'r') as yml_file:
        config = yaml.safe_load(yml_file)
    return config
