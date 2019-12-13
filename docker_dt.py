import os
from os.path import expanduser, join, abspath
import subprocess
import datetime
import shutil
import paramiko


class ExpRunner:

    def __init__(self, host_user_dir: str, docker_user_dir, docker_user,
                    docker_port,
                    script_path: str, script_args: str,
                    nodes: list, nGPU: str, eth: str, bw_limit: str, default_bw,
                    log_folder=None) -> None:
        """"""
        self.host_user_dir = host_user_dir
        self.docker_user_dir = docker_user_dir
        self.docker_user = docker_user
        self.docker_ssh_port = docker_port
        self.script_path = self._trans_docker_path(script_path)
        self.script_args = script_args
        self.nodes = nodes
        self.nGPU = nGPU # for each machine
        self.eth = eth # name if NIC
        self.bw_limit = bw_limit
        self.default_bw = default_bw
        self.log_folder = log_folder
        self.host_key = paramiko.RSAKey.from_private_key_file(expanduser("~/.ssh/id_rsa"))
        self.docker_key = paramiko.RSAKey.from_private_key_file("./DockerEnv/ssh-keys/id_rsa")
        self._init_host_ssh()
        
    def _trans_docker_path(self, path):
        return path.replace('~', self.docker_user_dir)

    def _init_host_ssh(self):
        print('='*10, 'initializing ssh connections')
        self.host_nodes = []
        for node in self.nodes:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(hostname=node, username="ubuntu", pkey=self.host_key)
            self.host_nodes.append((node, client))
            print('IP', node, 'DONE')
        print('='*10, 'initialization for ssh host node DONE')
    
    def _init_host_env(self):
        """"""
        for ip, cli in self.host_nodes:
            check_cmd = "mkdir ~/autorun; mkdir ~/autorun/horovod_logs; " \
                        "mkdir ~/autorun/horovod_logs/hooks; "\
                        "mkdir ~/autorun/horovod_logs/model_log; "\
                        "mkdir ~/autorun/horovod_logs/mpi_events"
            self._exec_cli_cmd(cli, check_cmd)
            check_cmd = "cd ~/autorun; ls|grep distributed-training"
            _, stdout, stderr = cli.exec_command(check_cmd)
            if stdout.read() != b"":
                git_pull = "cd ~/autorun/distributed-training; git pull"
                self._exec_cli_cmd(cli, git_pull, '{}: git pull'.format(ip))
            else:
                cmd = "cd ~/autorun;"\
                    "git clone https://github.com/zarzen/distributed-training.git"
                self._exec_cli_cmd(cli, cmd, "{}: clone training scripts".format(ip))
    
    def _exec_cli_cmd(self, cli, cmd, msg=None):
        if msg:
            print('>'*10, msg, '<'*10)
        _, stdout, stderr = cli.exec_command(cmd)
        print('cmd stdout: ', stdout.read().decode('utf-8'),
              "cmd stderr: ", stderr.read().decode('utf-8'))
        if msg:
            print('>'*10, 'DONE', msg, '<'*10)

    def _start_containers(self):
        """"""
        pull_cmd = "docker pull zarzen/horovod-mod:1.0"

        start_cmd = "docker run --gpus all --network=host --detach "\
            "-v {}/autorun/distributed-training:{}/distributed-training "\
            "-v {}/autorun/horovod_logs:{}/horovod_logs "\
            "zarzen/horovod-mod:1.0".format(self.host_user_dir, self.docker_user_dir,
                                            self.host_user_dir, self.docker_user_dir)
        self.docker_ids = {}
        for (ip, cli) in self.host_nodes:
            print('>'*10, ip, '<'*10)
            self._exec_cli_cmd(cli, pull_cmd, "{}: pull docker image".format(ip))
            _, stdout, stderr = cli.exec_command(start_cmd)
            _docker_id = stdout.read().decode('utf-8')
            self.docker_ids[ip] = _docker_id
            print('docker_id', _docker_id)
            print('Start Errors:', stderr.read().decode('utf-8'))
            print('='*10, ip, 'start container DONE', '='*10)

    def _kill_containers(self):
        """ after experiments done"""
        print('*'*10, 'killing docker containers')
        kill_cmd = "docker container kill {}"
        for ip, cli in self.host_nodes:
            if ip in self.docker_ids:
                self._exec_cli_cmd(cli, kill_cmd.format(self.docker_ids[ip]), ip)
        print('*'*10, 'kill containers done')

    def bandwith_control(self):
        """
        """
        del_cmd = "sudo tc qdisc del dev {} root tbf rate 40Gbit latency 400ms burst 3000kbit".format(self.eth)
        # if self.bw_limit = "" then we don't execute the add_cmd
        add_cmd = "sudo tc qdisc add dev {} root tbf rate {} latency 400ms burst 3000kbit".format(self.eth, self.bw_limit)
        for (ip, cli) in self.host_nodes:
            # try to delete rate limit
            self._exec_cli_cmd(cli, del_cmd, "{}: delete bandwidth limit".format(ip))
            # ensure limit deleted
            self._exec_cli_cmd(cli, del_cmd, "{}: delete bandwidth limit".format(ip)) 
            if self.bw_limit:
                self._exec_cli_cmd(cli, add_cmd, "{}: add bandwidth limit {}".format(ip, self.bw_limit))

    def exec_dist_train(self):
        """ execute distributed training script at rank0
        :return process:
        """
        train_cmd = self.build_train_cmd()
        print("Exec:", " ".join(train_cmd))

        # ssh into rank0 container
        ip, _ = self.host_nodes[0]
        rank0 = paramiko.SSHClient()
        rank0.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        rank0.connect(hostname=ip, port=self.docker_ssh_port, 
                        username=self.docker_user, 
                        pkey=self.docker_key) 

        _, stdout, stderr = rank0.exec_command(" ".join(train_cmd))
        print("-"*10, 'training log')
        print(stdout.read().decode('utf-8'))
        print(stderr.read().decode('utf-8'))
        print('-'*10, 'training log end')
    
    def build_train_cmd(self):
        """"""
        nNodes = len(self.nodes)
        np = str(nNodes * int(self.nGPU))
        hosts = ",".join(["{}:{}".format(ip, self.nGPU) for ip in self.nodes])
        cmd = ["NCCL_DEBUG=INFO",
                "HOROVOD_NUM_NCCL_STREAMS=4",
                "horovodrun", 
                "-np", np,
                "-H", hosts,
                "python3", 
                self.script_path, 
                self.script_args]
        return cmd
    
    def _get_logs(self):
        cpu_logs, net_logs = self._get_cpu_net_log()
        hook_logs, model_logs, mpi_logs = self._get_horovod_logs()
        return cpu_logs, net_logs, hook_logs, model_logs, mpi_logs
    
    def run(self):
        """"""

        print('initiating host env')
        self._init_host_env()

        self.exist_logs = self._get_logs()
        print('='*10, "working on bandwidth control")
        self.bandwith_control()
        print('='*10, "bandwidth control DONE")

        cpu_p, net_p = self._exe_res_monitor()
        print(">"*10, 'launched CPU & Network monitoring')

        print('='*10, 'Start containers', )
        self._start_containers()

        print('*'*10, 'Start working on experiment script')
        self.exec_dist_train()
        print('*'*10, 'Experiment finished')

        cpu_p.terminate()
        net_p.terminate()

        print('End experiment')
        self.move_log()
        

    def _exe_res_monitor(self):
        """ execute cpu and network bandwidth monitor
        """
        # record existing logs
        cpu_monitor_script = expanduser("~/autorun/monitor_cpu.py")
        net_monitor_script = expanduser("~/autorun/monitor_net.py")
        cpu_p = subprocess.Popen(["python3", cpu_monitor_script],
            stdout=subprocess.DEVNULL)
        net_p = subprocess.Popen(["python3", net_monitor_script],
            stdout=subprocess.DEVNULL)
        return cpu_p, net_p

    def move_log(self):
        """ rename horovod_logs -> horovod_logs_<bandwidth>,
        moving cpu.log and net.log into horovod_logs_<bandwidth> folder
        """
        # cpu, net, hook, model, mpi
        n_cpu, n_net, n_hook, n_model, n_mpi = self._get_logs()
        e_cpu, e_net, e_hook, e_model, e_mpi = self.exist_logs
        def _moving(src, dst, files):
            for _f in files:
                shutil.copy2(join(src, _f), join(dst, _f))
        dst_folder = self.log_folder if self.log_folder \
            else "./log_archives/{}-{}-{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), 
                self.bw_limit, self.default_bw)
        os.makedirs(dst_folder)
        _moving("./logs/cpu", dst_folder, n_cpu - e_cpu)
        _moving("./logs/net", dst_folder, n_net - e_net)
        _moving("./horovod_logs/hooks", dst_folder, n_hook-e_hook)
        _moving("./horovod_logs/model_log/", dst_folder, n_model-e_model)
        _moving("./horovod_logs/mpi_events", dst_folder, n_mpi-e_mpi)
        with open(join(dst_folder, "readme"), 'w+') as ofile:
            ofile.write("bandwidth limit: " + self.bw_limit)
    
    def _get_cpu_net_log(self):
        """ 
        record current exisiting logs
        """
        log_path = "./logs"
        log_path = expanduser(log_path)
        net_logs = os.listdir(join(log_path, 'net'))
        cpu_logs = os.listdir(join(log_path, 'cpu'))
        return set(cpu_logs), set(net_logs)
    
    def _create_horovod_logs_folder(self):
        base_dir = "./horovod_logs"
        if not os.path.exists(base_dir):
            os.makedirs('./horovod_logs')
        if not os.path.exists(join(base_dir, "hooks")):
            os.makedirs(join(base_dir, "hooks"))
        if not os.path.exists(join(base_dir, "model_log")):
            os.makedirs(join(base_dir, "model_log"))
        if not os.path.exists(join(base_dir, "mpi_events")):
            os.makedirs(join(base_dir, "mpi_events"))

    def _get_horovod_logs(self):
        base_dir = "./horovod_logs"
        hook_logs = os.listdir(join(base_dir, "hooks"))
        model_logs = os.listdir(join(base_dir, "model_log"))
        mpi_logs = os.listdir(join(base_dir, "mpi_events"))
        return set(hook_logs), set(model_logs), set(mpi_logs)

    def __del__(self):
        self._kill_containers()

def main():
    """"""
    exp = ExpRunner(host_user_dir="/home/ubuntu", docker_user_dir="/home/cluster", 
                    docker_user="cluster",
                    docker_port=2022,
                    script_path="~/distributed-training/test_scripts/pytorch_resnet50_cifar10.py", 
                    script_args="--epochs 1", # args of the script we want to run
                    nodes=["localhost", "172.31.29.187"], # list of worker's ip, the first one the rank0
                    nGPU="1", # nGPU on each machine
                    eth="ens3", # NIC interface name, used for bandwidth limit
                    bw_limit="", # limiting bandwidth, 100Mbit, 1Gbit, 10Gbit 25Gbit, 40Gbit,
                    default_bw="1.45Gbit", # default bandwidth
                    log_folder="" # if not specified, it will used the timestamp
                )
    exp.run()


if __name__ == "__main__":
    main()