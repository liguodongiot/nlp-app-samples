import hydra
from omegaconf import DictConfig, OmegaConf
import os

# 通过环境变量指定要使用哪个配置文件
env = os.getenv('DB_INFOS', 'pg')


# python main_hydra.py db.driver=root
@hydra.main(config_path="../conf", config_name=env, version_base=None)
def test(cfg: DictConfig):
    print(cfg.db.passwd)
    print(cfg.db.driver)
    print(cfg)
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    test()
