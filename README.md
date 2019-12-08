# Instructions

## `init_env.py`
The script is used for environment initialization. 
modify the `IPs` and `nccl_home` correspondingly
```python
    helper = Initializer(
        IPs = ['localhost', '172.31.29.187'],
        nccl_home = "/usr/local/nccl"
    )
```

## `dt_exp.py` distributed-training experiments runner
it will start `mpirun` and gather logs into `log_archives`
need to change nodes, nGPU, bw_limit, eth
```python
    exp = ExpRunner(python_bin, 
                "~/autorun/distributed-training/test_scripts/pytorch_resnet50_cifar10.py", 
                "--epochs 1", # args of the script we want to run
                ["localhost", "172.31.29.187"], # list of worker's ip
                nGPU="1", # nGPU on each machine
                eth="ens3", # NIC interface name, used for bandwidth limit
                bw_limit="1Gbit", # limiting bandwidth, 100Mbit, 1Gbit, 10Gbit 25Gbit, 40Gbit,
                log_folder="" # if not specified, it will used the timestamp
                )
```