from numpy import linalg 

def plot_grid_search_resuls(params_grid, grid_search_res, param_name_list, vmin=0.7, vmax=1):
    
    p1, p2, p3, p4 = params_grid
    fig, axs = plt.subplots(len(p1),len(p2), figsize=(8,6))
    
    y_len = len(p3)
    x_len = len(p4)
    x = np.arange(x_len)
    y = np.arange(y_len)
    grid_size = x_len * y_len
    
    if len(axs.shape)<2:
        axes = axs.ravel()
    
    for i, ax in enumerate(axes):
        # im = ax.imshow(grid_search[:,4].reshape(y_len ,x_len), vmin=0.7, vmax=1)
        im = ax.imshow(grid_search_res[grid_size*i:grid_size*(i+1),4].reshape(y_len ,x_len), vmin=vmin, vmax=vmax)
        ax.set_xticks(x, p4)
        ax.set_yticks(y, p3)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xlabel(param_name_list[4], fontsize=10)
        ax.set_ylabel(param_name_list[3], fontsize=10)

    cols = ['param_name_list[4]'+'={}'.format(col) for col in p4]
    rows = ['param_name_list[3]'+'={}'.format(row) for row in p3]

    for ax, col in zip(axs, cols):
        ax.set_title(col, fontsize=10, pad=40)

    for ax, row in zip(axs, rows):
        ax.set_ylabel(row, rotation=0, fontsize=10, labelpad=70)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.ax.tick_params(labelsize=10)

    # fig.tight_layout()
    plt.show()
    
    
def plot_cm(num_classes_learned, cm_results, obj_level=False, fig_size=4):
    plt.figure(figsize=(fig_size,fig_size))
    
    if obj_level:
        all_cls_names = np.array(train_ds.obj_names + ["Unknown"])
    else:
        all_cls_names = np.array(train_ds.cat_names + ["Unknown"])
    learned_cls_inds = nc_bm.classes_order[:num_classes_learned] + [n_classes]
    # +1's are for "Unknown" class
    num_classes_cm = num_classes_learned + 1
    
    # +1's are for "Unknown" class
    orig_cm = results["ConfusionMatrix_Stream/eval_phase/test_stream"].numpy()
    cm = np.zeros((num_classes_cm,num_classes_cm))
    print(cm.shape)

    cls_inds = [(f,s) for f in learned_cls_inds for s in learned_cls_inds]
    cm_inds =  [(f,s) for f in range(num_classes_cm) for s in range(num_classes_cm)]
    
    cm[tuple(zip(*cm_inds))]=orig_cm[tuple(zip(*cls_inds))]
    
    cm = cm[:-1,:] # remove "Unknown" on the rows
    cm = cm / linalg.norm(cm, axis=1, ord=1).reshape(num_classes_learned,1)
    
    plt.imshow(cm)
    # excldue "Unknown" on the rows
    plt.xticks(np.arange(num_classes_cm), all_cls_names[learned_cls_inds], fontsize=6, rotation = 90) 
    plt.yticks(np.arange(num_classes_learned), all_cls_names[learned_cls_inds[:-1]], fontsize=6)
    plt.tight_layout()