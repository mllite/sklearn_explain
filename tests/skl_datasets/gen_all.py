import os
from sklearn_explain.tests.skl_datasets import skl_datasets_test as skltest

def createDirIfNeeded(dirname):
    try:
        os.makedirs(dirname);
    except:
        pass


lClassDatasets =  skltest.define_tested_class_datasets();
lClassifiers = skltest.define_tested_classifiers();



def create_script(ds , model):
        print("GENERATING_MODEL" , model, ds);
        dirname = "tests/skl_datasets/" + str(ds) ;
        print(dirname);
        createDirIfNeeded(dirname);
        filename = dirname + "/skl_dataset_" + ds + "_" + model + "_code_gen.py";
        file = open(filename, "w");
        print("WRTITING_FILE" , filename);
        file.write("from sklearn_explain.tests.skl_datasets import skl_datasets_test as skltest\n");
        file.write("\n\n");
        args = "\"" + ds + "\" , \"" + model + "\"";
        file.write("skltest.test_class_dataset_and_model(" + args + ")\n");
        file.close();
    
# class
for ds in lClassDatasets:
    for model in lClassifiers:
        print("GENERATING_MODEL" , ds , model);
        create_script(ds, model)

