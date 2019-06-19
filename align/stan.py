"""For running the Stan model for illocutions."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

from align import util


def get_model_data(data, author_baselines, author_means, illocution_dict,
                   category_dict):
    model_data = {
        'num_dyads': len(data),
        'num_illocutions': len(illocution_dict),
        'num_categories': len(category_dict),
        'num_commenters': len(list(set([x['b_author'] for x in data]))),
        'num_observations': 0,
        'dyad_id': [],
        'illocution': [],
        'category': [],
        'commenter': [],
        'n_base': [],
        'c_base': [],
        'n_align': [],
        'c_align': [],
        'std_dev': 0.25
    }

    for x in data:
        for illocution in x['illocutions']:
            for category in category_dict.keys():
                # since we calculate baselines off other data, only look at
                # alignment observations here
                if x['a'][category] == 0.:
                    continue

                author = x['b_author']
                if author in author_baselines.keys():
                    baseline = author_baselines[author]
                else:
                    baseline = author_means
                model_data['n_base'].append(int(baseline[category]['n']))
                model_data['c_base'].append(int(baseline[category]['c']))

                model_data['num_observations'] += 1
                model_data['dyad_id'].append(x['_id'])
                model_data['illocution'].append(illocution_dict[illocution])
                model_data['category'].append(category_dict[category])
                model_data['commenter'].append(x['b_author'])

                model_data['n_align'].append(int(x['b_wc']))
                model_data['c_align'].append(int(round(
                    x['b_wc'] * x['b'][category] / 100, 0)))

    return model_data


def extract_samples(fit, param_name):
    samples = []
    n_chains = len(fit.sim['samples'])
    n_iter = fit.stan_args[0]['iter']
    warmup = fit.stan_args[0]['warmup']
    take = n_iter - warmup
    for ix in range(n_chains):
        samples += list(fit.sim['samples'][ix].chains[param_name][-take:])
    return samples


def get_posterior_samples(fit, illocution_dict):
    posteriors = {}
    rev_illocution_dict = util.rev_dict(illocution_dict)
    for i in range(len(illocution_dict)):
        illocution = rev_illocution_dict[i+1]
        param_name = 'eta_illocution_align[%s]' % (i + 1)
        posteriors[illocution] = extract_samples(fit, param_name)
    return posteriors


def plot_dists(posteriors):
    plt.figure(figsize=(6, 16))
    sns.set(style="darkgrid")
    data = {'illocution': [], 'alignment': []}
    for illocution in posteriors.keys():
        for sample in posteriors[illocution]:
            data['illocution'].append(illocution)
            data['alignment'].append(sample)
    df = pd.DataFrame(data=data)
    sns.catplot(x='alignment', y='illocution', data=df, kind='bar',
                ci=95)


def pairwise_t(posteriors):
    mat = np.zeros((len(posteriors), len(posteriors)))
    for i, ill_i in enumerate(posteriors.keys()):
        for j, ill_j in enumerate(posteriors.keys()):
            if i != j:
                mat[i, j] = round(stats.ttest_ind(
                    a=posteriors[ill_i],
                    b=posteriors[ill_j],
                )[1], 2)
    return mat


def compare_dists(posteriors, a, b, palette='husl'):
    data = {'Illocution': [], 'Alignment': [], '': []}
    for x in posteriors[a]:
        data['Illocution'].append(a)
        data['Alignment'].append(x)
        data[''].append('')
    for x in posteriors[b]:
        data['Illocution'].append(b)
        data['Alignment'].append(x)
        data[''].append('')
    df = pd.DataFrame(data=data)
    sns.set(style="whitegrid")
    sns.violinplot(x="", y="Alignment", hue="Illocution", data=df,
                   palette=palette, split=True)
    plt.savefig('compare_dists.png')
