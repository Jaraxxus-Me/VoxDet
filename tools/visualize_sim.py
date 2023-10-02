import seaborn as sns
import numpy as np
import os
from matplotlib import pyplot as plt

folder = 'geometry_gconv3/lmo'
obj_id = '1'

img_path = os.path.join(folder, 'obj_{}_sim'.format(obj_id))
os.makedirs(img_path, exist_ok=True)
sim_file = os.path.join(folder, 'obj_{}_sim.npz'.format(obj_id))
sim_data = np.load(sim_file, allow_pickle=True)

feat_sim = sim_data['feat_sim'].item()
loc_dis = sim_data['loc_dis'].item()

for k in feat_sim.keys():
    img_name = os.path.join(folder, 'obj_{}_sim'.format(obj_id), '{}.svg'.format(k))
    f_sim = feat_sim[k]
    l_dis = loc_dis[k]
    # mask = np.zeros_like(f_sim)
    # mask[np.triu_indices_from(mask)] = True

    fig, axes = plt.subplots(2, 1, figsize=(15, 40))
    fig.suptitle('Similarity and Dis Heatmap for Frame {}'.format(k))
    sns.heatmap(f_sim, vmin=0.0, square=True,  cmap="YlGnBu", ax=axes[0])
    axes[0].set_title('Feat Similarity')
    sns.heatmap(l_dis, vmax=100.0, square=True,  cmap="YlGnBu", ax=axes[1])
    axes[1].set_title('Geometry Distance')

    plt.savefig(img_name)
