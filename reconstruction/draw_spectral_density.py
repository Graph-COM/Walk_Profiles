import numpy as np
import matplotlib.pyplot as plt
import os.path as osp


# for ER graphs
walk_length = 20

#dataset_list = ['CELE', 'FIG', 'bio_drug-drug', 'bio_drug-target', 'bio_function-function', 'bio_protein-protein']
#dataset_list = ['code2']
dataset_list = ['CELE', 'FIG', 'bio_drug-drug', 'bio_drug-target', 'bio_function-function', 'bio_protein-protein',
                'ADO', 'ATC', 'EMA', 'HIG', 'USA', 'PB', 'snap-cite-hep', 'snap-epinion', 'snap-wiki-vote',
                    'code2-9k', 'resnet', 'transformer', 'er_n5000_p_1-n_c_1.0000_seed_0', 'er_n5000_p_1-n_c_2.0000_seed_0',
                'er_n5000_p_1-n_c_4.0000_seed_0', 'er_n5000_p_1-n_c_8.0000_seed_0', 'er_n5000_p_1-n_c_16.0000_seed_0',
                'citeseer', 'cora_ml', 'ogbn-arxiv', 'bert', 'mask_rcnn', 'alexnet', 'code2-20k', 'code2-36k']
dataset_name_list = ['Celegans', 'Figeys', 'Drug-drug', 'Drug-target', 'Biological functions', 'Protein-protein',
                     'Adolescent health', 'Air traffic control', 'Email', 'High-school', 'US airports', 'Political blogs',
                     'Arxiv HEP-TH', 'Epinion', 'Wikipedia vote', 'Ogbg-code2($N$=9k)', 'Resnet', 'Transformer', 'ER($d=1$)', 'ER($d=2$)',
                     'ER($d=4$)', 'ER($d=8$)', 'ER($d=16$)', 'Citeseer', 'Cora_ML', 'Ogbn-arxiv', 'Bert', 'Mask_RCNN', 'AlexNet',
                     'Ogbg-code2($N$=20k)', 'Ogbg-code2($N$=36k)']
dataset_labels = ['Biological', 'Biological', 'Biological', 'Biological', 'Biological', 'Biological', 'Social', 'Transport', 'Information', 'Social', 'Transport',
                  'Information', 'Citation', 'Social', 'Social', 'Program Graphs', 'Computational Graphs', 'Computational Graphs',
                  'Random Graphs', 'Random Graphs', 'Random Graphs', 'Random Graphs', 'Random Graphs', 'Citation',
                  'Citation', 'Citation', 'Computational Graphs', 'Computational Graphs', 'Computational Graphs',
                  'Program Graphs', 'Program Graphs']
colors = {'Biological': 'red', 'Social': 'blue', 'Transport': 'green', 'Information': 'purple', 'Citation':'brown', 'Program Graphs': 'orange', 'Computational Graphs': 'grey'}
file_paths = ['results/ip_%s_m%d_nbt0_norm1_lcpn1' % (dataset, walk_length) for dataset in dataset_list]

# load errors
densities = []
for file_path in file_paths:
    densities.append(np.load(osp.join(file_path, 'spectrum_dist.npy')))


fig, axs = plt.subplots(2, 4, figsize=(18, 8))
axs = axs.flatten()
labels_to_plot = ['Random Graphs', 'Biological', 'Social', 'Citation', 'Computational Graphs', 'Program Graphs', 'Transport', 'Information']
#colors = ['#4d4d4d', '#377eb8', '#e41a1c', '#4daf4a', '#ff7f00', '#984ea3', '#a65628']
#colors = ['#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594']
colors = ['#4d4d4d', '#377eb8', '#e41a1c', '#4daf4a', '#ff7f00', '#984ea3', '#17becf']

#file_path = 'results/spectrum_%s_nbt0_norm1_lcpn1' % dataset_list[0]
#q = np.load(osp.join(file_path, 'frequency.npy'))
q = np.array([j / 2 / (walk_length + 1) for j in range(len(densities[0]))])

#for i, label in enumerate(set(dataset_labels)):
for i, label in enumerate(labels_to_plot):
    row = i // 4  # Row index (0 or 1)
    col = i % 4  # Column index (0 to 3)
    densities_label = [e for e, l in zip(densities, dataset_labels) if l == label]
    names_label = [name for name, l in zip(dataset_name_list, dataset_labels) if l == label]
    #q = [i for i in range(1, len(densities_label[0])+2)]
    j = 0
    for d, l in zip(densities_label, names_label):
        axs[i].plot(q, d / sum(d), linewidth=2.5, label=l, color=colors[j])
        j += 1
    axs[i].set_title(label, fontsize=18,pad=10)
    axs[i].set_xlabel('Potential $q$', fontsize=18)
    axs[i].set_ylabel('Spectral density of $\widehat{\Phi}(%d, \cdot)$' % walk_length, fontsize=18)
    #axs[i].set_yscale('log')
    axs[i].legend(
    fontsize=13.5,  # Small and clean font
        loc='best',  # Let matplotlib pick best position
        frameon=True,  # No box around legend (very Nature Physics style)
    )
    if col != 0:
        axs[i].set_ylabel('')  # Optional: also remove y-axis label
        # X-axis: Only show for bottom row
    if row != 1:
        axs[i].set_xlabel('')  # Optional: also remove x-axis label
for ax in axs:
    ax.tick_params(axis='both', labelsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    #ax.set_xticks([0, 0.10, 0.2, 0.25])
# Adjust layout
plt.tight_layout(pad=1.0, w_pad=2.5, h_pad=2.5)  # Good spacing between plots
#plt.show()
plt.savefig('./figs/final/spectral_density_q_len20.pdf', format='pdf', bbox_inches='tight', dpi=300)
pass


