import os
myhost = os.uname()[1]
print(myhost)
if(("travis-job" in myhost) or os.getenv('CIRCLECI')):
    print("on travis-ci or circle-ci")
    os.system("tar cvfz last_logs.tar.gz logs/")
    os.system("curl -s -T last_logs.tar.gz curldu.mp")

