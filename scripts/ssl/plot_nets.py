import wandb
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import json
import pandas as pd

# api = wandb.Api()
# v3 = api.runs("nas_ssl/simclr_imagenet_models_evalv3")
# v3_names = [run.name for run in v3 if run.State=='finished']
# v2 = api.runs("nas_ssl/simclr_imagenet_models_evalv2")

# runs = list(v3) + [run for run in v2 if run.State=='finished' and (run.name not in v3_names or (run.name in v3_names and v3[v3_names.index(run.name)].State!='finished'))]


# names = [run.name for run in runs if 'epoch_top1_test' in run.summary]
# dataset = [name.split('-')[0].split('_')[-1] for name in names]
# models = [name.split('-')[1] for name in names]
# accs = [run.summary['epoch_top1_test'] for run in runs if 'epoch_top1_test' in run.summary]

# df = pd.DataFrame(list(zip(dataset, models, accs)), \
#                     columns=['Dataset', 'Model Name','Accuracy'])
# df.to_pickle('/vulcanscratch/sgirish/dummy/df.pickle')
# exit()
for key in ['mobilenet','resnet']:
# key = 'mobilenet'
    title = 'ResNet' if key == 'resnet' else 'MobileNet'

    with open(f'/vulcanscratch/sgirish/dummy/resnet_params.json','r') as f:
        net_params = json.load(f)
    with open(f'/vulcanscratch/sgirish/dummy/mobilenet_params.json','r') as f:
        mobilenet_params = json.load(f)

    net_params.update(mobilenet_params)
    dataset_titles = {'aircraft':'FGVC Aircraft', 'flower102':'Oxford Flowers', 'cars196':'Stanford Cars', 'imagenet':'ImageNet',
                        'cifar10':'CIFAR-10', 'cifar100':'CIFAR-100', 'cub200':'CUB', 'dogs120':'Stanford Dogs',
                        'mit67':'MIT67', 'sport8':'Sports', 'svhn':'SVHN'}

    df = pd.read_pickle('/vulcanscratch/sgirish/dummy/df.pickle')
    c_maxes = df.groupby(['Dataset', 'Model Name']).Accuracy.transform(max)
    df = df.loc[df['Accuracy'] == c_maxes]
    dataset_subset_resnet_accs = ['cifar100', 'mit67', 'sport8', 'cars196', 'aircraft']
    dataset_subset_mobilenet_accs = ['flower102', 'cifar10', 'mit67', 'cars196', 'sport8']
    dataset_subset_resnet_params = ['imagenet', 'cifar100', 'sport8', 'cars196', 'aircraft']
    dataset_subset_mobilenet_params = ['imagenet', 'flower102', 'cifar10', 'cars196', 'sport8']

    dataset_subset_accs = dataset_subset_resnet_accs if key == 'resnet' else dataset_subset_mobilenet_accs
    dataset_subset_params = dataset_subset_resnet_params if key == 'resnet' else dataset_subset_mobilenet_params
    # dataset_subset_resnet_accs = ['aircraft', 'flower102', 'cars196', 'cifar10', 'cifar100', 'cub200', 'dogs120', 'mit67', 'sport8', 'svhn']
    # dataset_subset_resnet_params = ['imagenet', 'flower102', 'cars196', 'cifar10', 'aircraft', 'cub200', 'dogs120', 'mit67', 'sport8', 'svhn']

    df_mobilenet_params = df[df['Dataset'].isin(dataset_subset_mobilenet_params)]
    df_resnet_params = df[df['Dataset'].isin(dataset_subset_resnet_params)]
    df_mobilenet_accs = df[df['Dataset'].isin(dataset_subset_mobilenet_accs)]
    df_resnet_accs = df[df['Dataset'].isin(dataset_subset_resnet_accs)]

    df_net_params = df_resnet_params if key == 'resnet' else df_mobilenet_params
    df_net_accs = df_resnet_accs if key == 'resnet' else df_mobilenet_accs

    df_accs_imagenet = df[df['Dataset']=='imagenet']
    imagenet_acc_dict = dict(zip(df_accs_imagenet['Model Name'], df_accs_imagenet['Accuracy']))
    max_params = 30 if key=='resnet' else 10
    min_params = 12 if key=='resnet' else 0
    min_accs = 55 if key=='resnet' else 0
    imagenet_acc_dict = {k:v for k,v in imagenet_acc_dict.items() if v>min_accs}
    nets = [k for k in df_resnet_params['Model Name'] if k in net_params and key in k and net_params[k]['params']<max_params \
            and (net_params[k]['bottleneck'] if key == 'resnet' else True) \
            and (imagenet_acc_dict[k]>min_accs if k in imagenet_acc_dict else True) and net_params[k]['params']>min_params]

    imagenet_acc_dict = {k:v for k,v in imagenet_acc_dict.items() if k in nets}
    nets = [k for k in nets if k in imagenet_acc_dict]
    imagenet_acc_full = df_net_accs[df_net_accs['Model Name'].isin(imagenet_acc_dict)]
    imagenet_acc_full['ImageNet Accuracy'] = [imagenet_acc_dict[model_name] for model_name in imagenet_acc_full['Model Name']]
    if key == 'resnet':
        imagenet_acc_full['Block'] = ['BottleNeck' if net_params[model_name]['bottleneck'] else 'BasicBlock' for model_name in imagenet_acc_full['Model Name']]
    imagenet_acc_full = imagenet_acc_full[imagenet_acc_full['ImageNet Accuracy']>min_accs]
    imagenet_param_full = df_net_params[df_net_params['Model Name'].isin(nets)]
    imagenet_param_full = imagenet_param_full
    imagenet_param_full['Params'] = [net_params[k]['params'] for k in imagenet_param_full['Model Name']]
    imagenet_param_full['Bottom Params'] = [net_params[k]['top_params'] for k in imagenet_param_full['Model Name']]
    imagenet_param_full['Top Params'] = [net_params[k]['bottom_params'] for k in imagenet_param_full['Model Name']]
    imagenet_param_full['Top/Bottom Param Ratio'] = [net_params[k]['bottom_params']/net_params[k]['top_params'] for k in imagenet_param_full['Model Name']]
    if key == 'resnet':
        imagenet_param_full['Block'] = ['BottleNeck' if net_params[k]['bottleneck'] else 'BasicBlock' for k in imagenet_param_full['Model Name']]
    imagenet_param_full = imagenet_param_full[imagenet_param_full['Model Name'].isin(nets)]
    # print(min(list(imagenet_param_full[imagenet_param_full['Dataset']=='imagenet']['Accuracy'])))
    plt.clf()
    plt.grid()
    num_cols = 5
    sns.set_style("darkgrid")
    sns.set(font_scale=1.4)
    # grid = sns.FacetGrid(df, col = "Dataset", col_wrap=5, sharey=False, sharex=True, col_order=[n for n in dataset_names if n!='imagenet'])
    # grid.map(sns.lmplot, "ImageNet Accuracy", "Accuracy")
    # grid.add_legend()

    corr = {'aircraft':-0.38, 'cars196':-0.3,'cifar10':0.85,'cifar100':0.91,'cub200':0.64,
            'dogs120':0.9,'flower102':0.38,'mit67':0.57,'sport8':-0.14,'svhn':0.44}
    corr_strings = {'aircraft':'-.38', 'cars196':'-.3','cifar10':'.85','cifar100':'.91','cub200':'.64',
            'dogs120':'.9','flower102':'.38','mit67':'.57','sport8':'-.14','svhn':'.44'}
    cur_dataset_titles = {k:title+f'-{dataset_titles[k]} ('+r'$r$='+corr_strings[k]+')' for k in dataset_subset_accs}
    g = sns.lmplot(x="ImageNet Accuracy", y="Accuracy", col="Dataset", col_order=dataset_subset_accs, fit_reg=True,
                data=imagenet_acc_full, col_wrap=5, sharey=False, sharex=True, truncate=False, ci=100, x_ci=100)

    in_min, in_max = min(list(imagenet_acc_dict.values())), max(list(imagenet_acc_dict.values()))
    for i in range(len(dataset_subset_accs)):
        dataset_name = dataset_subset_accs[i]
        # print(dataset_name, cur_accs.min(),cur_accs.max())
        cur_dataset_accs = list(imagenet_acc_full[imagenet_acc_full['Dataset']==dataset_name]['Accuracy'])
        g.axes[i].set_ylim((min(cur_dataset_accs)*0.998,max(cur_dataset_accs)*1.002))
        g.axes[i].set_xlim((in_min*0.998,in_max*1.002))
        g.axes[i].set_title(cur_dataset_titles[dataset_name])
        # g.axes[i].text(0,0, "An annotation", horizontalalignment='center', size='medium', color='black', weight='semibold')
        # g.axes[i].annotate(f'Linear corr. coefficient={corr[dataset_name]}', xy=(0.3,0.5), xycoords='figure fraction',xytext=(0.2,0.05), 
        #                     textcoords='axes fraction', fontsize=14)
    plt.show()
    plt.savefig(f'/vulcanscratch/sgirish/results_imagenet/downstream_accs_small_{key}.pdf', bbox_inches='tight')


    plt.clf()
    g = sns.lmplot(x="Params", y="Accuracy", col="Dataset", col_order=dataset_subset_params, fit_reg=True, order=2,
                data=imagenet_param_full, col_wrap=5, sharey=False, sharex=True, truncate=False, ci=100, x_ci=100)
    for i in range(len(dataset_subset_params)):
        dataset_name = dataset_subset_params[i]
        # print(dataset_name, cur_accs.min(),cur_accs.max())
        cur_dataset_accs = list(imagenet_param_full[imagenet_param_full['Dataset']==dataset_name]['Accuracy'])
        g.axes[i].set_ylim((min(cur_dataset_accs)*0.998,max(cur_dataset_accs)*1.002))
        # g.axes[i].set_xlim((in_min*0.998,in_max*1.002))
        g.axes[i].set_title(f"{title}-"+dataset_titles[dataset_name])
        # g.axes[i].text(0,0, "An annotation", horizontalalignment='center', size='medium', color='black', weight='semibold')
        # g.axes[i].annotate(f'Linear corr. coefficient={corr[dataset_name]}', xy=(0.3,0.5), xycoords='figure fraction',xytext=(0.2,0.05), 
        #                     textcoords='axes fraction', fontsize=14)
    plt.show()
    plt.savefig(f'/vulcanscratch/sgirish/results_imagenet/downstream_params_small_{key}.pdf', bbox_inches='tight')

    dataset_subset_resnet_top_params = ['imagenet', 'cars196', 'sport8']
    dataset_subset_mobilenet_top_params = ['imagenet', 'cars196', 'sport8']
    dataset_subset_top_params = dataset_subset_resnet_top_params if key == 'resnet' else dataset_subset_resnet_top_params

    norm = plt.Normalize(min(imagenet_param_full["Bottom Params"]), max(imagenet_param_full["Bottom Params"]))
    cmap = sns.color_palette("rocket_r")
    sm = plt.cm.ScalarMappable(cmap="rocket_r", norm=norm)
    sm.set_array([])
    sns.set(font_scale=1.6)
    # ax = sns.heatmap([[k] for k in list(imagenet_param_full['Bottom Params'])], cmap=cmap)
    # colorbar = ax.collections[0].colorbar
    plt.clf()
    g = sns.lmplot(x="Top/Bottom Param Ratio", y="Accuracy", hue="Bottom Params", col="Dataset", palette='rocket_r', col_order=dataset_subset_top_params, fit_reg=False, order=2,
                data=imagenet_param_full, col_wrap=len(dataset_subset_top_params), sharey=False, sharex=True, truncate=False, ci=100, x_ci=100, legend=False)
    for i in range(len(dataset_subset_top_params)):
        dataset_name = dataset_subset_top_params[i]
        # print(dataset_name, cur_accs.min(),cur_accs.max())
        cur_dataset_accs = list(imagenet_param_full[imagenet_param_full['Dataset']==dataset_name]['Accuracy'])
        g.axes[i].set_ylim((min(cur_dataset_accs)*0.998,max(cur_dataset_accs)*1.002))
        # g.axes[i].set_xlim((in_min*0.998,in_max*1.002))
        g.axes[i].set_title(f"{title}-"+dataset_titles[dataset_name])
        # g.axes[i].text(0,0, "An annotation", horizontalalignment='center', size='medium', color='black', weight='semibold')
        # g.axes[i].annotate(f'Linear corr. coefficient={corr[dataset_name]}', xy=(0.3,0.5), xycoords='figure fraction',xytext=(0.2,0.05), 
        #                     textcoords='axes fraction', fontsize=14)
    # plt.colorbar(list(imagenet_param_full['Bottom Params']))

    plt.colorbar(sm, label = 'Bottom params')
    # plt.colorbar(g, ax=g.axes)
    plt.show()
    plt.savefig(f'/vulcanscratch/sgirish/results_imagenet/downstream_params_top_bottom_{key}.pdf', bbox_inches='tight')