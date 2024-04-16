from config.base_config import Config
from model.GSM import GSE

class ModelFactory:
    @staticmethod
    def get_model(config: Config):
        if config.arch == 'GSE':
            return GSE(config)
        else:
            raise NotImplemented
