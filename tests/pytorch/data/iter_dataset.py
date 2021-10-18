
import torch
import math
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from torch.utils.data import get_worker_info


class MyIterableDataset(IterableDataset):
     def __init__(self, start, end):
         super(MyIterableDataset).__init__()
         assert end > start, "this example code only works with end >= start"
         self.start = start
         self.end = end

     def __iter__(self):
         worker_info = get_worker_info()
         print(f"worker_info: {worker_info}")
         if worker_info is None:  # single-process data loading, return the full iterator
             iter_start = self.start
             iter_end = self.end
         else:  # in a worker process
             # split workload
             per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
             worker_id = worker_info.id
             iter_start = self.start + worker_id * per_worker
             iter_end = min(iter_start + per_worker, self.end)
         return iter(range(iter_start, iter_end))



if __name__ == "__main__":
    # should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
    ds = MyIterableDataset(start=1, end=100)

    print(ds)

    dl = DataLoader(ds, num_workers=0)

    # Single-process loading
    print(list(dl))
    print("--------")
    print(list(dl))

    print("========")
    print(list(DataLoader(ds, num_workers=2)))
 


