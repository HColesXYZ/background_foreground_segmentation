import tensorflow_datasets as tfds

import bfseg.data.nyu.Nyu_depth_v2_labeled
import bfseg.data.nyu_subsampled
import bfseg.data.hive.bfseg_validation_labeled
import bfseg.data.hive.office_rumlang_validation_labeled
import bfseg.data.meshdist.bfseg_cla_meshdist_labels
import bfseg.data.pseudolabels
from datasets import load_dataset
#import os
#os.environ['CURL_CA_BUNDLE'] = "/etc/ssl/certs/ca-bundle.crt"

#load_dataset("sayakpaul/nyu_depth_v2")
#fds.load('OfficeRumlangValidationLabeled')
#tfds.load('MeshdistPseudolabels')
tfds.load('nyu_depth_v2')
#dataset = tfds.load(name='mnist')
tfds.load('nyu_subsampled')
#tfds.load("BfsegValidationLabeled")
#tfds.load("meshdist_pseudolabels")
#tfds.load("office_rumlang_validation_labeled")
#tfds.load('BfsegCLAMeshdistLabels')
