from hydra.utils import instantiate
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../conf_recursive", config_name="config", version_base=None)
def test(cfg: DictConfig):

    # 默认情况下，Hydra将递归实例化嵌套对象。
    trainer = instantiate(cfg.trainer)
    print(trainer)
    # Trainer(
    #  optimizer=Optimizer(algo=SGD,lr=0.01),
    #  dataset=Dataset(name=Imagenet, path=/datasets/imagenet)
    # )

    # 可以替代嵌套对象的参数
    trainer = instantiate(
        cfg.trainer,
        optimizer={"lr": 0.3},
        dataset={"name": "cifar10", "path": "/datasets/cifar10"},
    )
    print(trainer)
    # Trainer(
    #   optimizer=Optimizer(algo=SGD,lr=0.3),
    #   dataset=Dataset(name=cifar10, path=/datasets/cifar10)
    # )

    optimizer = instantiate(cfg.trainer, _recursive_=False)
    print(optimizer)
    # Trainer(
    #     optimizer={
    #         '_target_': 'my_app.Optimizer', 'algo': 'SGD', 'lr': 0.01
    #     },
    #     dataset={
    #         '_target_': 'my_app.Dataset', 'name': 'Imagenet', 'path': '/datasets/imagenet'
    #     }
    # )
    print("------")


if __name__ == "__main__":
    test()
