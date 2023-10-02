import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
sns.set_style("white")

plt.figure(figsize=(10,7), dpi= 80)
# pkl_path = 'work_dirs/val_gmconv3/val_record.npy'
# fig_path = 'work_dirs/val_gmconv3/val_record.png'
# # Plot
# # data on val set
# data = np.load(pkl_path, mmap_mode="r")
# n, bins, patches = plt.hist(x=data, bins=100, color='#0504aa',
#                             alpha=0.7, rwidth=0.85)
# plt.grid(axis='y', alpha=0.75)
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('My Very Own Histogram')
# p_le = (data<0.6).sum()/data.shape[0]
# p_ge = (data>0.6).sum()/data.shape[0]
# plt.savefig(fig_path)


folder = 'geometry_gconv3/ycbv'
obj_id = '2'

img_path = os.path.join(folder, 'obj_{}_sim_full.jpg'.format(obj_id))
# os.makedirs(img_path, exist_ok=True)
sim_file = os.path.join(folder, 'obj_{}_sim.npz'.format(obj_id))
sim_data = np.load(sim_file, allow_pickle=True)

feat_sim = sim_data['feat_sim'].item()
loc_dis = sim_data['loc_dis'].item()

data = []
c_dis = []

for k in feat_sim.keys():
    # img_name = os.path.join(folder, 'obj_{}_sim'.format(obj_id), '{}.svg'.format(k))
    f_sim = feat_sim[k]
    l_dis = loc_dis[k]
    # mask = np.zeros_like(f_sim)
    # mask[np.triu_indices_from(mask)] = True
    closest_ind = np.argmin(l_dis, axis=1)
    for i in range(f_sim.shape[0]):
        data.append(f_sim[i, closest_ind[i]])
        c_dis.append(l_dis[i, closest_ind[i]])

plt.scatter(x=c_dis, y=data, color='#0504aa', s=0.4)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Distance/mm')
plt.ylabel('Similarity')
plt.xlim(0, 10)
plt.title('Gemetric Verification of OBJ {}'.format(obj_id))
plt.savefig(img_path)