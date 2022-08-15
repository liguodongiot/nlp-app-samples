
from hydra.utils import instantiate
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="../conf_instantiation", config_name="config", version_base=None)
def test(cfg: DictConfig):
    opt = instantiate(cfg.optimizer)
    print(opt)
    # Optimizer(algo=SGD,lr=0.01)

    # override parameters on the call-site
    opt = instantiate(cfg.optimizer, lr=0.2)
    print(opt)
    # Optimizer(algo=SGD,lr=0.2)

    print("------")

if __name__ == "__main__":
    test()

