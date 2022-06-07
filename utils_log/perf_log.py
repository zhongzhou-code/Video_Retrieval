import logging
import logging.config
import yaml

with open("utils_log/logconf.yml", "r", encoding='utf-8') as file:
    dict_conf = yaml.safe_load(file)

logging.config.dictConfig(dict_conf)

root = logging.getLogger()
logger1 = logging.getLogger('logger1')
logger2 = logging.getLogger('logger2')

logger2.info("This is INFO of logger2 !!")