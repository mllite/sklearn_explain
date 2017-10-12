import pandas as pd
import numpy as np

from . import settings as conf

class cAbstractScoreExplainer:

    def __init__(self , clf, settings = None):
        self.mClassifier = clf
        if(settings is None):
            self.mSettings = conf.cScoreExplainerConfig()
        else:
            self.mSettings = settings        
        self.clear_data()

    def clear_data(self):
        self.mUsedFeatureNames = None
        self.mScoreQuantiles = None
        self.mFeatureQuantiles = None
        self.mScoreBinInterpolators = None
        self.mScoreBinInterpolationCoefficients = None
        self.mScoreBinInterpolationIntercepts = None
        self.mContribMeans = {}
        self.mExplanations = []
        self.mExplanationData = {}
        self.mCategories = {}

    # infer explanation data using the training dataset
    def fit(self, X):
        self.clear_data()
        if(self.mSettings.mFeatureNames is None):
            self.mFeatureNames_ = ['Feature_' + str(col+1) for col in range(X.shape[1])]
        else:
            self.mFeatureNames_ = self.mSettings.mFeatureNames
        self.create_score_stats(X)
        pass


    def explain(self, X):
        df = self.compute_reason_codes(X)
        return df

    def get_score(self, X):
        assert(0)

    def get_used_features(self, X):
        eps = 1e-10
        used_indices = []
        non_used_indices = []
        full_score = self.get_score(X)
        (NR , NC) = X.shape
        is_constant = np.abs(np.min(full_score) - np.max(full_score)) < eps
        print("CONST_SCORE_DETECTIOM", np.min(full_score), np.max(full_score) , is_constant)
        for col_idx in range(NC):
            # delete column i , put max + 1 everywhere (to be sure to change all the values)
            new_col = np.full((NR , 1) , np.max(X[:, col_idx]) + 1)
            X_without_col_idx = np.concatenate((X[:, 0:col_idx] , new_col , X[:, col_idx+1:]) , axis=1)
            partial_score = self.get_score(X_without_col_idx)
            # did the score change ?
            distance = np.linalg.norm(full_score - partial_score)
            if(distance > eps):
                used_indices.append(col_idx)
            else:
                non_used_indices.append(col_idx)
        if(is_constant):
            used_indices = [non_used_indices[0]]
            non_used_indices = non_used_indices[1:]
        lFeatures = self.get_feature_names()
        print("NON_USED_FEATURES" , [lFeatures[col_idx] for col_idx in  non_used_indices])
        print("USED_FEATURES" , [lFeatures[col_idx] for col_idx in  used_indices])
        return used_indices

    def getScoreQuantiles(self, col_data , bin_count):
        # Add the possibility to customize the binning of score and features #8
        # user defined ??
        lCustomScoreQuantiles = self.mSettings.mCustomScoreQuantiles
        quantiles = None
        if(lCustomScoreQuantiles is not None):
            quantiles = lCustomScoreQuantiles
            print("CUSTOM_SCORE_QUANTILES" , quantiles)
        else:
            quantiles = self.computeQuantiles(col_data, bin_count)
        return quantiles

    def getFeatureQuantiles(self, col_name, col_data , bin_count):
        # Add the possibility to customize the binning of score and features #8
        # user defined ??
        lCustomFeatureQuantiles = self.mSettings.mCustomFeatureQuantiles
        quantiles = None
        if(lCustomFeatureQuantiles is not None and lCustomFeatureQuantiles.get(col_name)):
            quantiles = lCustomFeatureQuantiles.get(col_name)
            print("CUSTOM_FEATURE_QUANTILES" , col_name, quantiles)
        else:
            quantiles = self.computeQuantiles(col_data, bin_count)
        return quantiles

    def computeQuantiles(self, col , bin_count):
        lBinCount = bin_count
        lBinCount = lBinCount if(lBinCount < (col.shape[0] / 30)) else int(col.shape[0] / 30)
        n_quantiles=[col.quantile(x/lBinCount) for x in range(1,lBinCount)]
        unique_quantiles = sorted(list(set(n_quantiles)))
        unique_quantiles = [-np.inf] + unique_quantiles
        q = pd.Series(unique_quantiles)
        quantiles = q.to_dict()
        quantiles[0] = -np.inf
        # print("QUANTILES_1" , col.name, bin_count, lBinCount, n_quantiles, unique_quantiles)
        # print("QUANTILES_2" , col.name, quantiles)
        return quantiles

    def computeCatgeories(self, col):
        categories = np.unique(col)
        # print("QUANTILES" , col.name, quantiles)
        return categories

    def get_bin_index(self, x, quantiles):
        qs = [k for k in quantiles.keys() if quantiles[k] <= x]
        res= max(qs) # if (len(qs) > 0) else max(quantiles.keys())
        # print("get_bin_index" , x , quantiles , res)
        return res

    def get_feature_names(self):
        return self.mFeatureNames_

    def get_explanations(self):
        return self.mExplanations

    def compute_RMSE(self , actual, predicted):
        n = len(predicted)
        rmse = np.linalg.norm(predicted - actual) / np.sqrt(n)
        return rmse
    
    def computeScoreInterpolators(self, df):
        from sklearn.linear_model import Ridge
        lExplanations = self.get_explanations()
        binned_features = [explain + '_encoded' for explain in lExplanations]
        self.mScoreBinInterpolators = {}
        self.mScoreBinInterpolationCoefficients = {}
        self.mScoreBinInterpolationIntercepts = {}
        for b in self.mScoreQuantiles.keys():
            bin_regression = Ridge(fit_intercept=False, solver='svd')
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
                pred = bin_regression.predict(bin_X)
                # print("PER_BIN_RMSE" , b , self.compute_RMSE(pred, bin_y))
                
        # print("SIGNIFICANT_BINS" , self.mScoreBinInterpolationCoefficients.keys())
        return df

    def get_feature_explanation_categorical(self, feat , value):
        output = "('" + feat + "' = " + str(value) + ")"
        return output

    def get_feature_explanation(self, feat , value):
        # value is a bin index (for the moment)
        (feat_min, feat_max) = self.get_bin_limits(value, self.mFeatureQuantiles[feat])
        output = "(" + str(feat_min) + " < '" + feat + "' <= " + str(feat_max) + ")"
            
        return output

    def get_explanation_human_friendly(self, explain , row):
        data = self.mExplanationData[explain]
        output = []
        for (i, feat_i) in enumerate(data):
            if(self.mSettings.is_categorical(feat_i)):
                output_i = self.get_feature_explanation_categorical(feat_i , row[feat_i + "_bin"])
            else:
                output_i = self.get_feature_explanation(feat_i , row[feat_i + "_bin"])
            output.append(output_i)
        return output

    def get_all_feature_combinations(self):
        lFeatures = self.mUsedFeatureNames
        import itertools
        lOrder = self.mSettings.mExplanationOrder
        if(len(lFeatures) > 50):
            lOrder = 1
        if(len(lFeatures) < lOrder):
            lOrder = len(lFeatures)
        lcombinations = itertools.combinations(lFeatures , lOrder)
        # print("COMBINATIONS" , [c for c in lcombinations])
        return lcombinations

    def generate_explanations(self, df):
        lFeatureEncoding = {}
        lFeatures = self.mUsedFeatureNames

        for col in lFeatures:
            if(self.mSettings.is_categorical(col)):
                df[col + '_bin'] = df[col]
                pass
            else:                
                if(self.mFeatureEncoding is None):
                    self.mFeatureQuantiles[col] = self.getFeatureQuantiles(col, df[col] , self.mSettings.mFeatureBins)       
                    print("FEATURE_QUANTILES" , col, self.mFeatureQuantiles[col])
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
                lFeatureEncoding[explain] = df[['Score' , explain]].groupby([explain])['Score'].quantile(0.5).to_dict()
                df[explain + '_encoded'] = df[explain].apply(lFeatureEncoding.get(explain).get)
            else:
                df[explain + '_encoded'] = df[explain].apply(self.mFeatureEncoding.get(explain).get)    
        if(self.mFeatureEncoding is None):
            self.mFeatureEncoding = lFeatureEncoding

    def create_score_stats(self, X):
        lScore = pd.Series(self.get_score(X))
        self.mScoreQuantiles = self.getScoreQuantiles(lScore , self.mSettings.mScoreBins)
        print("SCORE_QUANTILES" , self.mScoreQuantiles)
        lBinnedScore = lScore.apply(lambda x : self.get_bin_index(x , self.mScoreQuantiles))
        
        self.mFeatureQuantiles = {}
        self.mFeatureEncoding = None
        lFeatures = self.get_feature_names()
        df = pd.DataFrame(X , columns = lFeatures)
        df['Score'] = lScore
        df['BinnedScore'] = lBinnedScore

        lUsedFeatures_Indices = self.get_used_features(X)
        self.mUsedFeatureNames = [lFeatures[idx] for idx in lUsedFeatures_Indices]
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
        NC = min(len(self.mExplanations) , self.mSettings.mMaxReasons)
        reason_codes = reason_codes[:,0:NC]
        df_rc = pd.DataFrame(reason_codes, columns=['reason_' + str(NC-c) for c in range(NC)])
        df_rc = df_rc[list(reversed(df_rc.columns))]
        df_rc = pd.concat([df , df_rc] , axis=1)
        for c in range(NC):
            name = 'reason_' + str(c+1)
            lReason = df_rc[name]
            df_rc[name] = lReason.apply(lambda x : lExplanations[x])
            df_rc[name + "_idx"] = lReason
            lambda_detail = lambda row : self.get_explanation_human_friendly(lExplanations[row[name + "_idx"]] , row)
            df_rc['detailed_' + name] = df_rc.apply(lambda_detail, axis=1)
        # print(df_rc.sample(6, random_state=1960))
        return df_rc

    def get_explanation_values(self, explain):
        comb = self.mExplanationData[explain]
        lValues = []
        for feat in comb:
            feat_v = []
            for q in self.mFeatureQuantiles[feat].keys():
                feat_v.append(q)
            # print("get_explanation_values" , feat , tuple(feat_v))
            lValues = lValues + [feat_v]
        import itertools
        lprod = itertools.product(*lValues)
        return lprod      

    def get_bin_limits(self, bin_index, quantiles):
        bin_min = quantiles.get(bin_index)
        bin_max = quantiles.get(bin_index + 1 , np.inf)
        return (bin_min, bin_max)

    def get_local_score_card_from_score(self, score_value):
        """
        return a pandas dataframe giving the scorecard (union of all local scorecards)
        """
        lFeatures = self.mUsedFeatureNames
        rows = []
        NF = len(self.mExplanationData[self.mExplanations[0]])
        lExplanations = self.get_explanations()
        score_bin = self.get_bin_index(score_value, self.mScoreQuantiles)
        (score_min , score_max) = self.get_bin_limits(score_bin, self.mScoreQuantiles)
        for explain in lExplanations:
            comb = self.mExplanationData[explain]
            explain_key = "_".join(comb)
            vs = self.get_explanation_values(explain)
            for v in [k for k in vs]:
                row = []
                vi1 = []
                for (i,feat) in enumerate(comb):
                    row.append(feat)
                    if(self.mSettings.is_categorical(feat)):
                        (feat_min , feat_max) = (v[i] , v[i])
                    else:
                        (feat_min , feat_max) = self.get_bin_limits(v[i] , self.mFeatureQuantiles[feat])
                    row.append(feat_min)
                    row.append(feat_max)
                    vi1 = vi1 + [str(v[i])]
                coef = self.mScoreBinInterpolationCoefficients.get(score_bin).get(explain_key)
                intercept = self.mScoreBinInterpolationIntercepts.get(score_bin)
                vi2 = "_".join(vi1)
                # print(explain_key, vi2)
                # print(self.mFeatureEncoding)
                explain_encoded = self.mFeatureEncoding.get(explain_key).get(vi2 , 0.0)    
                lContrib = explain_encoded * coef
                row.append(lContrib)
                rows.append(row)
        assert(len(rows) > 0)
        sc_columns = [];
        for i in range(NF):
            sc_columns = sc_columns + ['feature_' + str(i+1) , 'feature_min_' + str(i+1) , 'feature_max_' + str(i+1)]
        sc_columns = sc_columns + ['points']
        score_card_df = pd.DataFrame(rows, columns=sc_columns)
    
        return score_card_df


    def get_local_score_cards(self):
        """
        return a dict of pandas dataframe giving the scorecard for each score bin 
        """
        result = {}
        for score_bin in self.mScoreQuantiles.keys():
            (score_min , score_max) = self.get_bin_limits(score_bin, self.mScoreQuantiles)
            score_card_df = self.get_local_score_card_from_score(score_max)
            result [str([score_min, score_max])] = score_card_df
        return result


    def get_local_score_card(self, X):
        lScore = self.get_score(X)
        if(self.mSettings.mDebug):
            print("GET_LOCAL_SCORE_CARD" , X , lScore)
        df = self.get_local_score_card_from_score(lScore)
        return df
        

