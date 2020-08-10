from __future__ import print_function

import torch
import numpy as np

def main():
    a = np.array([1, 2])
    b = np.array([8, 9])
    print(a + b)

    a = torch.tensor([1, 2])
    b = torch.tensor([8, 9])
    print(a + b)

    ones = torch.rand(3, 2)
    print(ones)
    print(ones.t())

    print(torch.cuda.is_initialized())

if __name__ == "__main__":
    main()