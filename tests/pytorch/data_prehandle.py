import torch
import torch.utils.data as Data

BATCH_SIZE = 5

def show_batch(loader):
    for epoch in range(3):
        for step, (batch_x, batch_y) in enumerate(loader):
            # training
            print("step:{}, batch_x:{}, batch_y:{}".format(step, batch_x, batch_y))


if __name__ == '__main__':
    x = torch.linspace(1, 10, 10)
    y = torch.linspace(10, 1, 10)

    print("~~~~~~~~~~~~~~~~~~~~~~")
    print(x, y)
    print("----------------------")

    # 把数据放在数据库中
    torch_dataset = Data.TensorDataset(x, y)
    

    loader = Data.DataLoader(
        # 从数据库中每次抽出batch size个样本
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1,
    )
    
    show_batch(loader)
