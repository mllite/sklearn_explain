from sklearn_explain.tests.skl_datasets_reg import skl_datasets_test as skltest


lExplainer = skltest.test_reg_dataset_and_model("RandomReg_10" , "Ridge_6")
print(lExplainer.mImplementation.__dict__)

print(lExplainer.mImplementation.mUsedFeatureNames)
print(lExplainer.mImplementation.mScoreQuantiles)
print(lExplainer.mImplementation.mFeatureQuantiles)
print(lExplainer.mImplementation.mScoreBinInterpolators)
print(lExplainer.mImplementation.mScoreBinInterpolationIntercepts)
print(lExplainer.mImplementation.mContribMeans)
print(lExplainer.mImplementation.mExplanations)
print(lExplainer.mImplementation.mExplanationData)
print(lExplainer.mImplementation.mCategories)

coeffs = lExplainer.mImplementation.mScoreBinInterpolationCoefficients
for k in coeffs:
    print(k, coeffs[k])


for k in lExplainer.mImplementation.mFeatureQuantiles:
    print(k, lExplainer.mImplementation.mFeatureQuantiles[k])

