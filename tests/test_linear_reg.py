from sklearn_explain.tests.skl_datasets_reg import skl_datasets_test as skltest


lExplainer = skltest.test_reg_dataset_and_model("RandomReg_10" , "Ridge_6")
print(lExplainer.mImplemantation.__dict__)

print(lExplainer.mImplemantation.mUsedFeatureNames)
print(lExplainer.mImplemantation.mScoreQuantiles)
print(lExplainer.mImplemantation.mFeatureQuantiles)
print(lExplainer.mImplemantation.mScoreBinInterpolators)
print(lExplainer.mImplemantation.mScoreBinInterpolationIntercepts)
print(lExplainer.mImplemantation.mContribMeans)
print(lExplainer.mImplemantation.mExplanations)
print(lExplainer.mImplemantation.mExplanationData)
print(lExplainer.mImplemantation.mCategories)

coeffs = lExplainer.mImplemantation.mScoreBinInterpolationCoefficients
for k in coeffs:
    print(k, coeffs[k])


for k in lExplainer.mImplemantation.mFeatureQuantiles:
    print(k, lExplainer.mImplemantation.mFeatureQuantiles[k])

