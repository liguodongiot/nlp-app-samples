import hydra
from omegaconf import DictConfig, OmegaConf
import os


# 如果新增配置项，需要加上+
# python main_hydra_group.py db=mysql db.db.driver=guodong +db.db.timeout=20

# python main_hydra_group.py --multirun db=mysql,pg
@hydra.main(config_path="../conf_group", config_name="config", version_base=None)
def test(cfg: DictConfig):
    # print(cfg.db.passwd)
    # print(cfg.db.driver)
    print(cfg)
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    test()
