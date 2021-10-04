from logconf import logging
import sys
import datetime

log = logging.getLogger('test_logger')
log.setLevel(logging.DEBUG)
log.info('this is a test')

print(str(sys.argv))
