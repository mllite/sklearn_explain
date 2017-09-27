import os
from sklearn_explain.tests.skl_datasets_reg import skl_datasets_test as skltest

def createDirIfNeeded(dirname):
    try:
        os.makedirs(dirname);
    except:
        pass


lRegDatasets =  skltest.define_tested_reg_datasets();
lRegifiers = skltest.define_tested_regressors();



def create_script(ds , model):
        print("GENERATING_MODEL" , model, ds);
        dirname = "tests/skl_datasets_reg/" + str(ds) ;
        print(dirname);
        createDirIfNeeded(dirname);
        filename = dirname + "/skl_dataset_" + ds + "_" + model + "_code_gen.py";
        file = open(filename, "w");
        print("WRTITING_FILE" , filename);
        file.write("from sklearn_explain.tests.skl_datasets_reg import skl_datasets_test as skltest\n");
        file.write("\n\n");
        args = "\"" + ds + "\" , \"" + model + "\"";
        file.write("skltest.test_reg_dataset_and_model(" + args + ")\n");
        file.close();
    
# class
for ds in lRegDatasets:
    for model in lRegifiers:
        print("GENERATING_MODEL" , ds , model);
        create_script(ds, model)

