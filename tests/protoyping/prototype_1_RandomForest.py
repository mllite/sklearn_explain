from sklearn import datasets
import pandas as pd

# %matplotlib inline

ds = datasets.load_breast_cancer();
NC = 4
lFeatures = ds.feature_names[0:NC]

df_orig = pd.DataFrame(ds.data[:,0:NC] , columns=lFeatures)
df_orig['TGT'] = ds.target
df_orig.sample(6, random_state=1960)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=120, random_state = 1960)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_orig[lFeatures].values, 
                                                    df_orig['TGT'].values, 
                                                    test_size=0.2, 
                                                    random_state=1960)

df_train = pd.DataFrame(X_train , columns=lFeatures)
df_train['TGT'] = y_train
df_test = pd.DataFrame(X_test , columns=lFeatures)
df_test['TGT'] = y_test

clf.fit(X_train , y_train)

###########################################

from sklearn.linear_model import *
def create_score_stats(df, feature_bins = 4 , score_bins=30):
    df_binned = df.copy()
    df_binned['Score'] = clf.predict_proba(df[lFeatures].values)[:,0]
    df_binned['Score_bin'] = pd.qcut(df_binned['Score'] , score_bins, labels=False, duplicates='drop')

    for col in lFeatures:
        df_binned[col + '_bin'] = pd.qcut(df[col] , feature_bins, labels=False, duplicates='drop')
    
    binned_features = [col + '_bin' for col in lFeatures]
    lInterpolted_Score= pd.Series(index=df_binned.index)
    bin_classifiers = {}
    coefficients = {}
    for b in range(score_bins):
        bin_clf = Ridge(random_state = 1960)
        bin_indices = (df_binned['Score_bin'] == b)
        # print("PER_BIN_INDICES" , b , bin_indices)
        bin_data = df_binned[bin_indices]
        bin_X = bin_data[binned_features]
        bin_y = bin_data['Score']
        if(bin_y.shape[0] > 0):
            bin_clf.fit(bin_X , bin_y)
            bin_classifiers[b] = bin_clf
            bin_coefficients = dict(zip(lFeatures, [bin_clf.coef_.ravel()[i] for i in range(len(lFeatures))]))
            print("PER_BIN_COEFFICIENTS" , b , bin_coefficients)
            coefficients[b] = bin_coefficients
            predicted = bin_clf.predict(bin_X)
            lInterpolted_Score[bin_indices] = predicted

    df_binned['Score_interp'] = lInterpolted_Score 
    return (df_binned , bin_classifiers , coefficients)

(df_cross_stats , per_bin_classifiers , per_bin_coefficients) = create_score_stats(df_train , feature_bins=20 , score_bins=20)
df_cross_stats.sample(6, random_state=1960)

###########################


df2 = df_cross_stats.sort_values('Score').reset_index()
print(df2.columns)
df2.plot('Score', ['Score_bin'])
df2.plot('Score', ['Score_interp' ])
for col in lFeatures:
    df2.plot('Score', [col + '_bin'])
df2.sample(12)


####################################

pd.crosstab(df_cross_stats['mean radius_bin'], df_cross_stats['Score_bin'])


#######################################

for col in lFeatures:
    lcoef = df_cross_stats['Score_bin'].apply(lambda x : per_bin_coefficients.get(x).get(col))
    lContrib = lcoef * df_cross_stats[col + '_bin']
    df1 = pd.DataFrame();
    df1['contrib'] = lContrib
    df1['Score_bin'] = df_cross_stats['Score_bin']
    lContribMeans = df1.groupby('Score_bin')['contrib'].mean().to_dict()
    print(lContribMeans)
    df_cross_stats[col + '_Effect'] = lContrib - df_cross_stats['Score_bin'].apply(lambda x : lContribMeans.get(x))

df_cross_stats.sample(6, random_state=1960)

#######################################

import numpy as np
reason_codes = np.argsort(df_cross_stats[[col + '_Effect' for col in lFeatures]].values, axis=1)
df_rc = pd.DataFrame(reason_codes, columns=['reason_' + str(NC-c) for c in range(NC)])
df_rc = df_rc[list(reversed(df_rc.columns))]
df_rc = pd.concat([df_cross_stats , df_rc] , axis=1)
for c in range(NC):
    df_rc['reason_' + str(c+1)] = df_rc['reason_' + str(c+1)].apply(lambda x : lFeatures[x])
print(df_rc.sample(6, random_state=1960))


###########################################


df_rc[['reason_' + str(NC-c) for c in range(NC)]].describe()

