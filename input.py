
import sys, os
import subprocess
import shlex
from subprocess import Popen

# INPUT_FILES = ["./nla.csv"]
NUM_STATES = [1,2,3,4,5,6,7,8,9,10]
NUM_THREADS = [32,64,128,256]

for j in range(len(NUM_STATES)):
	for k in range(len(NUM_THREADS)):
		var1 = str(NUM_STATES[j])
		var2 = str(NUM_THREADS[k])
		os.system("./data.sh %s %s" % (var1, var2))

# for i in INPUT_FILES:
	# for j in NUM_STATES:
	# 	for k in NUM_THREADS:
		# var1 = INPUT_FILES[i]
		# var2 = str(NUM_STATES[j])
		# var3 = str(NUM_THREADS[k])
		# os.system("./data.sh %s %s %s" % (var1, var2))

# Process=Popen(['./data.sh %s %s' % (str(NUM_STATES[j]),str(NUM_THREADS[k])], shell=True)
# subprocess.call(shlex.split('./data.sh INPUT_FILES[i] NUM_STATES[j] NUM_THREADS[k]'))
