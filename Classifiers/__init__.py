
# print "in init!"
import numpy
import logging



LOGNAME = "classifiers"
# logging.basicConfig(#filename=LOGNAME ,
#                             filemode='a'
#                             ,format='%(asctime)s %(message)s (%(levelname)s)'# %(name)s
#                             ,datefmt='%H:%M:%S'
#                             ,level=logging.DEBUG)
#
# # logging.info("Init()")
# # logging.warn("Init()")
# # logging.error("Init()")
# strm = logging.StreamHandler()
# # logger.addHandler(strm)
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger(LOGNAME)
rootLogger.level = logging.DEBUG


fileHandler = logging.FileHandler("{0}.log".format(LOGNAME))
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)


rootLogger.warn("a warning")
# logger.error("some error")
# logger.info("some info")
