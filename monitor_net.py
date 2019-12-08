import psutil
import time
from datetime import datetime
import os

def comp_bandwidth(new, old, interval):
    dRecv = new.bytes_recv - old.bytes_recv
    dSend = new.bytes_sent - old.bytes_sent

    return dRecv*8 / interval, dSend*8 / interval

def create_logfile():
    log_folder = "./logs/net"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    dt = datetime.fromtimestamp(time.time())
    timestamp = dt.strftime("%Y%m%d-%H%M%S")
    logfile = timestamp + '-net.log'
    return os.path.join(log_folder, logfile)

def main():
    interval = 0.1
    old_net_stat = psutil.net_io_counters()
    logfile = create_logfile()
    with open(logfile, 'w') as net_log:
        
        while True:
            new_net_stat = psutil.net_io_counters()
            bandwidth = comp_bandwidth(new_net_stat, old_net_stat, interval)
            print(time.time(), 'Receiving(bit/sec)', bandwidth[0], 'Sending(bit/sec)', bandwidth[1])
            net_log.write("{}, {}, {}\n".format(time.time(), bandwidth[0], bandwidth[1]))

            old_net_stat = new_net_stat
            time.sleep(interval)
            net_log.flush()

if __name__ == "__main__":
    main()