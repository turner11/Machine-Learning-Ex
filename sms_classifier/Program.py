from Preprocessor import Preprocessor
import random as rnd
import pandas as pd
import numpy as np
#import pylab as plt
import os

LBL_HAM = "ham"
LBL_SPAM = "spam"
DATA_PATH = "SMSSpamCollection"

#dataFolder = "C:\Dev\Jupiter\Task\\"
#dataPath = dataFolder+"SMSSpamCollection"
#print dataPath
#os.chdir(dataFolder)

pp = Preprocessor()
crps_ham = pp.init_lists(DATA_PATH,LBL_HAM)
crps_spam = pp.init_lists(DATA_PATH,LBL_SPAM)

print "Ham: "+crps_ham[0]
print "Spam: "+crps_spam[0]

tok_ham = pp.batch_tokeniz(crps_ham)
tok_spam = pp.batch_tokeniz(crps_spam)

print "Ham: " + " ".join(tok_ham[0])
print "Spam: " + " ".join(tok_spam[0])