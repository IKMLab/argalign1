from sklearn import linear_model, model_selection, metrics
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from align import util


wanted = [
    'non-argumentative',
    'asserting', 'arguing',
    'disagreeing', 'agreeing', 'conceding',
    'questioning-rhetorical', 'questioning-assertive', 'questioning-pure',
    'challenging-pure', 'challenging-assertive', 'challenging-rhetorical',
    'disagreeing with extra argument', 'disagreeing without extra argument',
    'agreeing with extra argument', 'agreeing without extra argument',
    'conceding and disagreeing',
    'conceding without disagreeing',
    'conceding and agreeing'
]


def get_x_align(data, category_dict, author_baselines, author_means):
    x = []

    def get_x_vec(_x):
        baselines = author_baselines[_x['b_author']]
        vec = np.ones((1, len(baselines))) * -5
        for category in baselines.keys():
            b_eta = baselines[category]['eta']
            ix = category_dict[category] - 1
            if _x['a'][category] > 0.:
                n = int(_x['b_wc'])
                c = int(round(_x['b'][category] / 100 * n, 0))
                p = c / n if c and n else 1e-10
                eta = util.logit(p)
                vec[0, ix] = eta - b_eta
                vec = np.clip(vec, a_min=-5., a_max=5.)
        return vec

    for d in data:
        if d['b_author'] not in author_baselines.keys():
            author_baselines[d['b_author']] = author_means
        v = get_x_vec(d)
        x.append(v)

    x = np.concatenate(x, axis=0)

    return x


def get_y(data, illocution_dict):

    def get_y_vec(_x):
        vec = np.zeros((1, len(illocution_dict)))
        for illocution in _x['annotations']['Tim']:
            ix = illocution_dict[illocution] - 1
            vec[0, ix] = 1
        return vec

    y = []
    for x in data:
        y.append(get_y_vec(x))
    y = np.concatenate(y, axis=0)

    return y


def evaluate_all(X_train, Y_train, X_test, Y_test, illocution_dict):
    results = {}
    rev_illocution_dict = util.rev_dict(illocution_dict)
    with tqdm(total=len(rev_illocution_dict)) as pbar:
        for i in range(len(rev_illocution_dict)):
            ill = rev_illocution_dict[i+1]
            if ill not in wanted:
                pbar.update()
                continue
            y_train = Y_train[:, i]
            y_test = Y_test[:, i]
            if only_one_class(y_train) or only_one_class(y_test):
                pbar.update()
                continue
            results[ill] = evaluate(X_train, y_train, X_test, y_test)
            pbar.update()
    return results


def evaluate(x_train, y_train, x_test, y_test, folds=5):
    rocs = []
    for _ in range(20):
        cv = model_selection.GridSearchCV(
            estimator=linear_model.LogisticRegression(
                solver='lbfgs',
                max_iter=10000),
            param_grid={
                'C': [0.1, 0.5, 1.]
            },
            scoring='roc_auc',
            cv=folds,
            iid=True)
        cv.fit(x_train, y_train)
        estimator = cv.best_estimator_
        y_pred = estimator.predict(x_test)
        try:
            rocs.append(metrics.roc_auc_score(y_test, y_pred))
        except:
            rocs.append(0.5)
    return np.mean(rocs)


def only_one_class(y, n_splits=5):
    s = np.sum(y)
    return s < n_splits or s == y.shape[0]


def compare_results(a, b):
    for ill in a.keys():
        print('%4.3f\t%4.3f\t%4.3f\t%s'
              % (a[ill], b[ill], b[ill] - a[ill], ill))


def plot_comparison(a, b):
    dd = {'Illocution': [], 'Change in ROC AUC': []}
    for ill in wanted:
        if ill in a.keys():
            dd['Illocution'].append(ill)
            dd['Change in ROC AUC'].append(b[ill] - a[ill])
    df = pd.DataFrame(data=dd)
    plt.figure(figsize=(4, 8))
    sns.set()
    sns.boxplot(x='Change in ROC AUC', y='Illocution', data=df)
