import json
import yaml
import numpy as np
import pandas as pd
import pickle


# To JSON
def save_to_json(data, filename="ga_results.json"):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# To YAML
def save_to_yaml(data, filename="ga_results.yaml"):
    with open(filename, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


# Population to pickle
def save_population(population, filename="ga_population.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(population, f, protocol=pickle.HIGHEST_PROTOCOL)

# Save Normalized front
def save_norm(norm, filename="ga_norm.tsv"):
    np.savetxt(fname=filename, X=norm, delimiter='\t')