
import sys, os
import subprocess
import shlex
from subprocess import Popen

INPUT_FILES = ["./nla.csv", "bacteria.csv"]
NUM_STATES = [1,2,3,4,5,6,7,8,9,10]
NUM_THREADS = [32,64,128,256]

def add_quotes(s):
	s = '"{}"'.format(s)
	return "'%s'" % s

for i in range(len(INPUT_FILES)):
	for j in range(len(INPUT_FILES)):
		for k in range(len(INPUT_FILES)):
			var1 = str(add_quotes(INPUT_FILES[i]))
			var2 = str(NUM_STATES[j])
			var3 = str(NUM_THREADS[k])

			os.system("./data.sh %s %s %s" % (var1, var2, var3))
