import numpy as np
import matplotlib.pyplot as plt
import os.path as osp


# for ER graphs
walk_length = 50

#dataset_list = ['CELE', 'FIG', 'bio_drug-drug', 'bio_drug-target', 'bio_function-function', 'bio_protein-protein']
#dataset_list = ['code2']
dataset_list = ['CELE', 'FIG', 'bio_drug-drug', 'bio_drug-target', 'bio_function-function', 'bio_protein-protein',
                'ADO', 'ATC', 'EMA', 'HIG', 'USA', 'PB', 'snap-cite-hep', 'snap-epinion', 'snap-wiki-vote',
                    'code2-9k', 'resnet', 'transformer', 'er_n5000_p_1-n_c_1.0000_seed_0', 'er_n5000_p_1-n_c_2.0000_seed_0',
                'er_n5000_p_1-n_c_4.0000_seed_0', 'er_n5000_p_1-n_c_8.0000_seed_0', 'er_n5000_p_1-n_c_16.0000_seed_0',
                'citeseer', 'cora_ml', 'ogbn-arxiv', 'bert', 'mask_rcnn', 'alexnet', 'code2-20k', 'code2-36k']
dataset_name_list = ['Celegans', 'Figeys', 'Drug-drug', 'Drug-target', 'Bio-functions', 'Protein-protein',
                     'Adolescent health', 'Air traffic control', 'Email', 'High-school', 'US airports', 'Political blogs',
                     'Arxiv HEP-TH', 'Epinion', 'Wikipedia vote', 'Ogbg-code2($N$=9k)', 'Resnet', 'Transformer', 'ER($d=1$)', 'ER($d=2$)',
                     'ER($d=4$)', 'ER($d=8$)', 'ER($d=16$)', 'Citeseer', 'Cora_ML', 'Ogbn-arxiv', 'Bert', 'Mask_RCNN', 'AlexNet',
                     'Ogbg-code2($N$=20k)', 'Ogbg-code2($N$=36k)']
dataset_labels = ['Biological', 'Biological', 'Biological', 'Biological', 'Biological', 'Biological', 'Social', 'Transport', 'Information', 'Social', 'Transport',
                  'Information', 'Citation', 'Social', 'Social', 'Program Graphs', 'Computational Graphs', 'Computational Graphs',
                  'Random Graphs', 'Random Graphs', 'Random Graphs', 'Random Graphs', 'Random Graphs', 'Citation',
                  'Citation', 'Citation', 'Computational Graphs', 'Computational Graphs', 'Computational Graphs',
                  'Program Graphs', 'Program Graphs']
colors = {'Biological': 'red', 'Social': 'blue', 'Transport': 'green', 'Information': 'purple', 'Citation':'brown' , 'Program Graphs': 'orange', 'Computational Graphs': 'grey',
          'Random Graphs': 'black'}


markers = {'Biological':'o', 'Social': 's', 'Transport': '^', 'Information': 'v', 'Citation': 'D', 'Program Graphs':'P', 'Computational Graphs': 'X',
           'Random Graphs': '*'}
file_paths = ['results/ip_%s_m%d_nbt0_norm1_lcpn1' % (dataset, walk_length) for dataset in dataset_list]

# load errors
errors = []
for file_path in file_paths:
    errors.append(np.load(osp.join(file_path, 'error.npy')))


file_paths = ['results/spectrum_%s_nbt0_norm1_lcpn1' % dataset for dataset in dataset_list]

spectrum = []
for file_path in file_paths:
    spectrum.append(np.load(osp.join(file_path, 'largest_eigenvalues.npy')))
q = np.load(osp.join(file_path, 'frequency.npy'))
#plt.plot([i for i in range(2, len(errors[0])-2)], errors[0][0:-4], marker='o')


#fig, axs = plt.subplots(1, 2, figsize=(22, 8))
fig, axs = plt.subplots(1, 1, figsize=(10, 8))
#axs = axs.flatten()


# spectrum v.s. q on all graphs
#for i in range(len(spectrum)):
#    color = colors[dataset_labels[i]]
    #plt.plot(q, spectrum[i], color=color, label=dataset_labels[i])
#    axs[0].plot(q, spectrum[i], color=color, label=dataset_labels[i], alpha=0.8)
#axs[0].set_xlabel('Potential $q$', fontsize=18)
#axs[0].set_ylabel('Spectral Radius of $\widehat{A}_q$', fontsize=18)
#axs[0].set_ylim(0.6, 1)
#axs[0].tick_params(axis='both', labelsize=16)  # Change 14 to whatever you want
#handles, labels = axs[0].get_legend_handles_labels()
# Remove duplicates while preserving order
#unique = {}
#for h, l in zip(handles, labels):
#    if l not in unique:
#        unique[l] = h
#axs[0].legend(unique.values(), unique.keys(), fontsize=12)

#axs[0].legend(unique.values(), unique.keys(), fontsize=16,  # Small and clean font
#              loc='best',  # Let matplotlib pick best position
#              frameon=True,  # No box around legend (very Nature Physics style)
#              )
#plt.show()
#exit(0)

eval_pos = 8
errors_at_pos = [e[eval_pos] for e in errors]
#freq_at_pos = [s[eval_pos] for s in spectrum]
#freq_power_at_pos = [s[eval_pos]**walk_length for s in spectrum]
show_mode = 'one' # 'one', 'one-power', 'sum-power'
if show_mode == 'one':
    freq_at_pos = [s[eval_pos] for s in spectrum]
elif show_mode == 'one-power':
    freq_at_pos = [s[eval_pos] ** walk_length for s in spectrum]
elif show_mode == 'sum-power':
    freq_at_pos = [(s[eval_pos:] ** walk_length).sum() / (s[:eval_pos] ** walk_length).sum() for s in
                               spectrum]

#plt.figure(figsize=(10, 7.5))
#plt.scatter(freq_at_pos, errors_at_pos)
for i in range(len(dataset_list)):
    color = colors[dataset_labels[i]]
    marker = markers[dataset_labels[i]]
    #plt.scatter(freq_at_pos[i], errors_at_pos[i], facecolors='none', edgecolors=color, label=dataset_labels[i], s=120, marker=marker, linewidths=2)
    axs.scatter(freq_at_pos[i], errors_at_pos[i], facecolors='none', edgecolors=color, label=dataset_labels[i], s=120, marker=marker, linewidths=2)
axs.set_yscale("log")
#plt.xscale("log")
axs.set_xlabel('Spectral Radius of $\widehat{A}_{q_{%d}}$' % (eval_pos+2), fontsize=18)
axs.set_ylabel('Recon. errors of $\widehat{\Phi}(%d, \cdot)$ with top-%d qs' % (walk_length, eval_pos+2), fontsize=18)
#plt.title('%d-Walks Compress. error (%d qs) v.s. largest eigenvalues (%d-th q)' % (walk_length, eval_pos, eval_pos+1))
#plt.show()


from scipy.stats import pearsonr, spearmanr
sorted_indices = np.argsort(freq_at_pos)
X = np.array(freq_at_pos)[sorted_indices]
Y = np.log(np.array(errors_at_pos)[sorted_indices])
res = pearsonr(X, Y)
r_value = res.statistic
CI = res.confidence_interval(0.95)
m, b = np.polyfit(X, Y, 1)
axs.plot(X, np.exp(m*np.array(X) + b), color='black', linestyle='--', linewidth=1.5, label=f"$r={r_value:.2f}$(95%CI: ${CI[0]:.2f}$ to ${CI[1]:.2f}$)")
axs.tick_params(axis='both', labelsize=16)  # Change 14 to whatever you want


# Retrieve current legend handles and labels
handles, labels = plt.gca().get_legend_handles_labels()

# Remove duplicates while preserving order
unique = {}
for h, l in zip(handles, labels):
    if l not in unique:
        unique[l] = h

axs.legend(unique.values(), unique.keys(), fontsize=16,  # Small and clean font
        loc='best',  # Let matplotlib pick best position
        frameon=True,  # No box around legend (very Nature Physics style)
           )
#plt.show()
#plt.legend()
plt.savefig('./figs/final/recon_error_spectral_radius_len50.pdf', format='pdf', bbox_inches='tight', dpi=300)