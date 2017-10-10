import sys
import os
from tinydb import TinyDB, Query
from shutil import copy

# TINYDB_PATH = './tf-models/dcoral_result.json'
TINYDB_PATH = './result.json'
print os.path.abspath(os.path.dirname(TINYDB_PATH))
# copy for backup
print 'doing backup of result json file...'
copy(TINYDB_PATH, TINYDB_PATH[:-5] + '(backup).json')


db = TinyDB(TINYDB_PATH)
hptb = db.table('lba-hyper-params')
hp = Query()

is_empty = lambda s: not os.path.exists(os.path.join(os.path.abspath(os.path.dirname(TINYDB_PATH)), s))

attribute = hp.logdir # change this for another name for dir path

print 'found %d empty hyper params. deleting...' %\
      len(hptb.search(attribute.test(is_empty)))

hptb.remove(attribute.test(is_empty))

print 'done.'