import logging
import logging.config
from pathlib import Path

import yaml

from snepits import data as data
from snepits import features as features
from snepits import models as models
from snepits import visualization as visualization

# Define project base directory
project_dir = Path(__file__).resolve().parents[1]

# Define log output locations
info_out = str(project_dir / 'info.log')
error_out = str(project_dir / 'errors.log')

# Read log config file
with open(project_dir / 'logging.yaml', 'rt') as f:
    config = yaml.safe_load(f.read())
logging.config.dictConfig(config)

# Define module logger
logger = logging.getLogger(__name__)
