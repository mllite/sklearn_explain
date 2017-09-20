import os
import glob

subdirs = ["tests/protoyping"] +  glob.glob("tests/skl_datasets/*")


all_target = "";
for subdir1 in subdirs:
    test_target = "";
    for filename in glob.glob(subdir1 + "/*.py"):
        bn1 = os.path.basename(filename);
        lKeep = (not bn1.startswith('gen_all.py') and not bn1.startswith('gen_makefile.py'))
        bn = subdir1 + "/" + bn1;
        bn2 = bn.replace("/" , "_")
        logfile = bn.replace("/" , "_");
        logfile = "logs/" + logfile.replace(".py" , ".log");
        if(lKeep):
            print(bn1 + ": " , "\n\t", "-python3" , filename , " > " , logfile , " 2>&1");
        test_target = bn1 + " " + test_target;

    subdir1_label = subdir1.replace("/" , "_")
    print(subdir1_label + ":" , test_target)
    all_target = all_target + " " + subdir1_label;

#print("\n# ********************************************** \n");

print("all:  " , all_target, "\n\t\n");


