import pandas as pd
import numpy as np

class cClassificationModel_ScoreExplainer:

    def __init__(self , clf):
        self.mFeatureNames = None
        self.mScoreBins = 20
        self.mFeatureBins = 10
        self.mMaxReasons = 20
        self.mScoreQuantiles = None
        self.mFeatureQuantiles = None
        self.mScoreBinInterpolators = None
        self.mScoreBinInterpolationCoefficients = None
        self.mScoreBinInterpolationIntercepts = None
        self.mContribMeans = {}
        self.mClassifier = clf
        self.mDebug = True

    # infer explanation data using the training dataset
    def fit(self, X):
        if(self.mFeatureNames is None):
            self.mFeatureNames_ = ['Feature_' + str(col+1) for col in range(X.shape[1])]
        else:
            self.mFeatureNames_ = self.mFeatureNames
        self.create_score_stats(X)
        pass


    def explain(self, X):
        df = self.compute_reason_codes(X)
        return df

    def get_score(self, X):
        if(hasattr(self.mClassifier , 'predict_proba')):
            if(self.mDebug):
                print("USING_PROBABILITY_AS_SCORE")
            return self.mClassifier.predict_proba(X)[:,0]
        if(hasattr(self.mClassifier , 'decision_function')):
            if(self.mDebug):
                print("USING_DECISION_FUNCTION_AS_SCORE")
            lDecision = self.mClassifier.decision_function(X)
            if(len(lDecision.shape) == 1):
                # binary classifier : RidgeClassifier and SGDClassifier
                return lDecision
            return lDecision[:,0]
        return None

    def computeQuantiles(self, col , bin_count):
        q = pd.Series(range(0,bin_count)).apply(lambda x : col.quantile(x/bin_count))
        quantiles = q.to_dict()
        # print("QUANTILES" , col.name, quantiles)
        return quantiles

    def get_bin_index(self, x, quantiles):
        res= min(quantiles.keys(), key=lambda y:float(quantiles[y]-x))
        # print("get_bin_index" , x , quantiles , res)
        return res

    def get_feature_names(self):
        return self.mFeatureNames_

    def computeScoreInterpolators(self, df):
        from sklearn.linear_model import Ridge
        lFeatures = self.get_feature_names()
        binned_features = [col + '_bin' for col in lFeatures]
        lInterpolted_Score= pd.Series(index=df.index)
        self.mScoreBinInterpolators = {}
        self.mScoreBinInterpolationCoefficients = {}
        self.mScoreBinInterpolationIntercepts = {}
        for b in self.mScoreQuantiles.keys():
            bin_regression = Ridge(random_state = 1960)
            bin_indices = (df['BinnedScore'] == b)
            # print("PER_BIN_INDICES" , b , bin_indices)
            bin_data = df[bin_indices]
            bin_X = bin_data[binned_features]
            bin_y = bin_data['Score']
            if(bin_y.shape[0] > 0):
                bin_regression.fit(bin_X , bin_y)
                self.mScoreBinInterpolators[b] = bin_regression
                bin_coefficients = dict(zip(lFeatures, [bin_regression.coef_.ravel()[i] for i in range(len(lFeatures))]))
                # print("PER_BIN_COEFFICIENTS" , b , bin_coefficients)
                self.mScoreBinInterpolationCoefficients[b] = bin_coefficients
                self.mScoreBinInterpolationIntercepts[b] = bin_regression.intercept_
                predicted = bin_regression.predict(bin_X)
                lInterpolted_Score[bin_indices] = predicted
        df['Score_interp'] = lInterpolted_Score
        # print("SIGNIFICANT_BINS" , self.mScoreBinInterpolationCoefficients.keys())
        return df

    def create_score_stats(self, X):
        lScore = pd.Series(self.get_score(X))
        self.mScoreQuantiles = self.computeQuantiles(lScore , self.mScoreBins)
        lBinnedScore = lScore.apply(lambda x : self.get_bin_index(x , self.mScoreQuantiles))
        
        self.mFeatureQuantiles = {}
        lFeatures = self.get_feature_names()
        df = pd.DataFrame(X , columns = lFeatures)
        df['Score'] = lScore
        df['BinnedScore'] = lBinnedScore
        for col in lFeatures:
            self.mFeatureQuantiles[col] = self.computeQuantiles(df[col] , self.mFeatureBins)       
            df[col + '_bin'] = df[col].apply(lambda x : self.get_bin_index(x , self.mFeatureQuantiles[col]))

        df = self.computeScoreInterpolators(df)
        df = self.estimate_effects(df)
        return df

    def estimate_effects(self, df):
        lFeatures = self.get_feature_names()
        self.mContribMeans = {}
        for col in lFeatures:
            coef = df['BinnedScore'].apply(lambda x : self.mScoreBinInterpolationCoefficients.get(x).get(col))
            intercept = df['BinnedScore'].apply(lambda x : self.mScoreBinInterpolationIntercepts.get(x))
            lContrib = coef * df[col + '_bin'] + intercept / len(lFeatures)
            df1 = pd.DataFrame();
            df1['contrib'] = lContrib
            df1['BinnedScore'] = df['BinnedScore']
            self.mContribMeans[col] = df1.groupby(['BinnedScore'])['contrib'].mean().to_dict()
            contrib_mean = df['BinnedScore'].apply(lambda x : self.mContribMeans.get(col).get(x))
            # print("MEAN_CONTRIB" , col, self.mContribMeans[col])
            df[col + '_Effect'] = lContrib - contrib_mean
        return df

    def compute_effects(self, df):
        lFeatures = self.get_feature_names()
        for col in lFeatures:
            coef = df['BinnedScore'].apply(lambda x : self.mScoreBinInterpolationCoefficients.get(x).get(col))
            intercept = df['BinnedScore'].apply(lambda x : self.mScoreBinInterpolationIntercepts.get(x))
            lContrib = coef * df[col + '_bin'] + intercept / len(lFeatures)
            contrib_mean = df['BinnedScore'].apply(lambda x : self.mContribMeans.get(col).get(x))
            df[col + '_Effect'] = lContrib - contrib_mean
        return df

    def get_contrib(self , col_name , score_bin , col_bin):
        coefficiens = self.mScoreBinInterpolationCoefficients.get(score_bin)
        coef = coefficiens.get(col_name)
        intercept = self.mScoreBinInterpolationIntercepts.get(score_bin)
        lContrib = lcoef * col_bin + intercept / len(coefficiens)
        return lContrib

    def compute_reason_codes(self, X):
        lScore = pd.Series(self.get_score(X))
        lBinnedScore = lScore.apply(lambda x : self.get_bin_index(x , self.mScoreQuantiles))
        
        lFeatures = self.get_feature_names()
        df = pd.DataFrame(X , columns = lFeatures)
        df['Score'] = lScore
        df['BinnedScore'] = lBinnedScore
        for col in lFeatures:
            df[col + '_bin'] = df[col].apply(lambda x : self.get_bin_index(x , self.mFeatureQuantiles[col]))

        df = self.compute_effects(df)
        reason_codes = np.argsort(df[[col + '_Effect' for col in lFeatures]].values, axis=1)
        NC = min(len(self.mFeatureNames_) , self.mMaxReasons)
        reason_codes = reason_codes[:,0:NC]
        df_rc = pd.DataFrame(reason_codes, columns=['reason_' + str(NC-c) for c in range(NC)])
        df_rc = df_rc[list(reversed(df_rc.columns))]
        df_rc = pd.concat([df , df_rc] , axis=1)
        for c in range(NC):
            df_rc['reason_' + str(c+1)] = df_rc['reason_' + str(c+1)].apply(lambda x : lFeatures[x])
        # print(df_rc.sample(6, random_state=1960))
        return df_rc
