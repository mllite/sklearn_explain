import pandas as pd
import numpy as np

class cClassificationModel_ScoreExplainer:

    def __init__(self , clf):
        self.mFeatureNames = None
        self.mScoreBins = 5
        self.mFeatureBins = 5
        self.mMaxReasons = 5
        self.mScoreQuantiles = None
        self.mFeatureQuantiles = None
        self.mScoreBinInterpolators = None
        self.mScoreBinInterpolationCoefficients = None
        self.mScoreBinInterpolationIntercepts = None
        self.mContribMeans = {}
        self.mClassifier = clf
        self.mExplanations = []
        self.mExplanationData = {}
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
            lProba = self.mClassifier.predict_proba(X)[:,1]
            lProba = lProba.clip(0.00001 , 0.99999)
            lOdds = lProba / (1.0 - lProba)
            return np.log(lOdds)
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
        lBinCount = bin_count
        lBinCount = lBinCount if(lBinCount < (col.shape[0] / 30)) else int(col.shape[0] / 30)
            
        q = pd.Series(range(0,lBinCount)).apply(lambda x : col.quantile(x/lBinCount))
        quantiles = q.to_dict()
        # print("QUANTILES" , col.name, quantiles)
        return quantiles

    def get_bin_index(self, x, quantiles):
        res= min(quantiles.keys(), key=lambda y:abs(float(quantiles[y]-x)))
        # print("get_bin_index" , x , quantiles , res)
        return res

    def get_feature_names(self):
        return self.mFeatureNames_

    def get_explanations(self):
        return self.mExplanations

    def computeScoreInterpolators(self, df):
        from sklearn.linear_model import Ridge
        lExplanations = self.get_explanations()
        binned_features = [explain + '_encoded' for explain in lExplanations]
        self.mScoreBinInterpolators = {}
        self.mScoreBinInterpolationCoefficients = {}
        self.mScoreBinInterpolationIntercepts = {}
        for b in self.mScoreQuantiles.keys():
            bin_regression = Ridge(fit_intercept=False, random_state = 1960)
            bin_indices = (df['BinnedScore'] == b)
            # print("PER_BIN_INDICES" , b , bin_indices)
            bin_data = df[bin_indices]
            bin_X = bin_data[binned_features]
            bin_y = bin_data['Score']
            if(bin_y.shape[0] > 0):
                bin_regression.fit(bin_X , bin_y)
                self.mScoreBinInterpolators[b] = bin_regression
                bin_coefficients = dict(zip(lExplanations, [bin_regression.coef_.ravel()[i] for i in range(len(lExplanations))]))
                # print("PER_BIN_COEFFICIENTS" , b , bin_coefficients)
                self.mScoreBinInterpolationCoefficients[b] = bin_coefficients
                self.mScoreBinInterpolationIntercepts[b] = bin_regression.intercept_
        # print("SIGNIFICANT_BINS" , self.mScoreBinInterpolationCoefficients.keys())
        return df

    def get_feature_explanation(self, feat , value):
        # value is a bin index (for the moment)
        q = self.mFeatureQuantiles[feat][value]
        if(value == 0):
            output = "('" + feat + "' <= " + str(q) + ")"
        else:
            prev_q = self.mFeatureQuantiles[feat][value - 1]
            output = "(" + str(prev_q) + " < '" + feat + "' <= " + str(q) + ")"
            
        return output

    def get_explanation_human_friendly(self, explain , values):
        data = self.mExplanationData[explain]
        output = []
        for (i, feat_i) in enumerate(data):
            output_i = self.get_feature_explanation(feat_i , values[i])
            output.append(output_i)
        return output

    def get_all_feature_combinations(self):
        lFeatures = self.get_feature_names()
        import itertools
        lcombinations = itertools.combinations(lFeatures , 2)
        # print("COMBINATIONS" , [c for c in lcombinations])
        return lcombinations

    def generate_explanations(self, df):
        lFeatureEncoding = {}
        lFeatures = self.get_feature_names()

        for col in lFeatures:
            if(self.mFeatureEncoding is None):
                self.mFeatureQuantiles[col] = self.computeQuantiles(df[col] , self.mFeatureBins)       
            df[col + '_bin'] = df[col].apply(lambda x : self.get_bin_index(x , self.mFeatureQuantiles[col]))


        combinations = self.get_all_feature_combinations()
        for comb in combinations:
            explain = "_".join(comb)
            if(self.mFeatureEncoding is None):
                self.mExplanations.append(explain)
                self.mExplanationData[explain] = comb
            cols = [c + "_bin" for c in comb]
            df[explain] = df[cols].apply(lambda row: "_".join([str(row[c]) for c in cols]), axis=1)
            if(self.mFeatureEncoding is None):
                lFeatureEncoding[explain] = df[['Score' , explain]].groupby([explain])['Score'].mean().to_dict()
                df[explain + '_encoded'] = df[explain].apply(lFeatureEncoding.get(explain).get)
            else:
                df[explain + '_encoded'] = df[explain].apply(self.mFeatureEncoding.get(explain).get)    
        if(self.mFeatureEncoding is None):
            self.mFeatureEncoding = lFeatureEncoding

    def create_score_stats(self, X):
        lScore = pd.Series(self.get_score(X))
        self.mScoreQuantiles = self.computeQuantiles(lScore , self.mScoreBins)
        lBinnedScore = lScore.apply(lambda x : self.get_bin_index(x , self.mScoreQuantiles))
        
        self.mFeatureQuantiles = {}
        self.mFeatureEncoding = None
        lFeatures = self.get_feature_names()
        df = pd.DataFrame(X , columns = lFeatures)
        df['Score'] = lScore
        df['BinnedScore'] = lBinnedScore

        self.generate_explanations(df);
        df = self.computeScoreInterpolators(df)
        df = self.estimate_effects(df)
        return df

    def estimate_effects(self, df):
        lExplanations = self.get_explanations()
        self.mContribMeans = {}
        for explain in lExplanations:
            coef = df['BinnedScore'].apply(lambda x : self.mScoreBinInterpolationCoefficients.get(x).get(explain))
            intercept = df['BinnedScore'].apply(lambda x : self.mScoreBinInterpolationIntercepts.get(x))
            lContrib = coef * df[explain + '_encoded'] + intercept / len(lExplanations)
            df1 = pd.DataFrame();
            df1['contrib'] = lContrib
            df1['BinnedScore'] = df['BinnedScore']
            self.mContribMeans[explain] = df1.groupby(['BinnedScore'])['contrib'].mean().to_dict()
            contrib_mean = df['BinnedScore'].apply(lambda x : self.mContribMeans.get(explain).get(x))
            # print("MEAN_CONTRIB" , col, self.mContribMeans[col])
            df[explain + '_Effect'] = lContrib - contrib_mean
        return df

    def compute_effects(self, df):
        lExplanations = self.get_explanations()
        for explain in lExplanations:
            coef = df['BinnedScore'].apply(lambda x : self.mScoreBinInterpolationCoefficients.get(x).get(explain))
            intercept = df['BinnedScore'].apply(lambda x : self.mScoreBinInterpolationIntercepts.get(x))
            lContrib = coef * df[explain + '_encoded'] + intercept / len(lExplanations)
            contrib_mean = df['BinnedScore'].apply(lambda x : self.mContribMeans.get(explain).get(x))
            df[explain + '_Effect'] = lContrib - contrib_mean
        return df

    def get_contrib(self , explain_name , score_bin , col_bin):
        coefficiens = self.mScoreBinInterpolationCoefficients.get(score_bin)
        coef = coefficiens.get(explain_name)
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

        self.generate_explanations(df)

        df = self.compute_effects(df)
        lExplanations = self.get_explanations()
        reason_codes = np.argsort(df[[explain + '_Effect' for explain in lExplanations]].values, axis=1)
        NC = min(len(self.mExplanations) , self.mMaxReasons)
        reason_codes = reason_codes[:,0:NC]
        df_rc = pd.DataFrame(reason_codes, columns=['reason_' + str(NC-c) for c in range(NC)])
        df_rc = df_rc[list(reversed(df_rc.columns))]
        df_rc = pd.concat([df , df_rc] , axis=1)
        for c in range(NC):
            df_rc['reason_' + str(c+1)] = df_rc['reason_' + str(c+1)].apply(lambda x : lExplanations[x])
            df_rc['deatiled_reason_' + str(c+1)] = df_rc['reason_' + str(c+1)].apply(lambda x : self.get_explanation_human_friendly(lExplanations[x] , [1 for c in lExplanations]))
        # print(df_rc.sample(6, random_state=1960))
        return df_rc
