# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: default
#     language: python
#     name: python3
# ---

# %%
import sys
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))

from utils.neuron_bag import NeuronBag
from utils.movie_maker import generate_tiling_movie_json
from utils.metric_functions import get_completeness_metrics

from utils import olc_client
c = olc_client.connect(verbose=True)

# %%
tiling_types = ['Dm4', 'Dm20', 'l-LNv', 'MeVP10']
roi_str = 'ME(R)'

# %%
text_dict = {}

for neuron_type in tiling_types: 

    instance = f"{neuron_type}_R"
    
    neuron_text_dict = {}
    df = get_completeness_metrics(instance=instance)

    n1 = df[df['roi']=='ME(R)'].loc[0, 'coverage_factor_trim']
    neuron_text_dict['coverage_factor'] = f"{n1:.2f}"
    n2 = df[df['roi']=='ME(R)'].loc[0, 'col_completeness']
    neuron_text_dict['columns_compl']= f"{n2:.2f}"
    n3 = df[df['roi']=='ME(R)'].loc[0, 'area_completeness']
    neuron_text_dict['area_compl']= f"{n3:.2f}"
    
    text_dict[neuron_type] = neuron_text_dict

# %%
params = {
    'Dm4': {
        'scale_start': 1.1
      , 'duration_text': 11.5
      , 'adv_time_roi_in': 0.75
      , 'adv_time_id_in': 1.5
      , 'adv_time_neigh_in': 3.0
      , 'adv_time_all_in': 5.74
    }
  , 'Dm20': {
        'scale_start': 1.2
      , 'duration_text': 11.75
      , 'adv_time_roi_in': 0.75
      , 'adv_time_id_in': 1.5
      , 'adv_time_neigh_in': 3.0
      , 'adv_time_all_in': 5.74
    }
  , 'l-LNv': {
        'scale_start': 1.2
      , 'duration_text': 8.5
      , 'adv_time_roi_in': 0.75
      , 'adv_time_id_in': 2.0
      , 'adv_time_neigh_in': 2.0
      , 'adv_time_all_in': 3.0
    }
  , 'MeVP10': {
        'scale_start': 1.0
      , 'duration_text': 12.0
      , 'adv_time_roi_in': 0.75
      , 'adv_time_id_in': 2.0
      , 'adv_time_neigh_in': 3.0
      , 'adv_time_all_in': 5.44
    }
}

# %%
for neuron_type in tiling_types:
    print(neuron_type)
    a_bag = NeuronBag(cell_type=neuron_type)
    a_bag.sort_by_distance_to_star()
    sorted_body_ids = a_bag.get_body_ids(a_bag.size)

    if a_bag.size < 10:
        num_neigh = int(a_bag.size/2)
    else:
        num_neigh = 10

    generate_tiling_movie_json(
        neuron_type=neuron_type
      , sorted_body_ids=sorted_body_ids
      , text_dict=text_dict
      , params=params
      , template="MEi_tiling.json.jinja"
      , number_of_neighbors=num_neigh
    )
    print(f"Json generation done for {neuron_type}")


# %%
