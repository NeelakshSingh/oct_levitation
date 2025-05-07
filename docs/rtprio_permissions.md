# Running Nodes at Real-Time Priority

All python controller nodes benefit highly from being run at a very high frequency. It is paramount at times in order to get sufficiently high control frequency with less computation times which will only add to pure input-output delays. Therefore, follow these steps to be able to run all nodes at real-time priority without requiring root privileges and without needing any special scheduler setting lines of code in the nodes themselves which is especially cumbersome for python nodes. (ChatGPT might be able to do in with a single prompt though)

Note that this guide is solely written for Ubuntu based systems, and following these steps will ensure that the system never takes resources away from critical control nodes but to ensure hard real-time guarantees you will need to use a kernel with real-time preemption a.k.a PREEMPT-RT. Also, some might think that paswordless sudo is an easy solution but you will run into issues with souring ROS as root which will make things complicated since the root profile files for terminals are better left untouched.

### STEP 1: Check if real-time priority is even possible

For running processes at real-time priority the `SCHED_FIFO` and `SCHED_RR` scheduling policies are required. To check if your system supports these policies, run the following command:

```bash
chrt --max
```

The output should list `SCHED_FIFO` and `SCHED_RR` and should have a maximum priority of 99. If they are not listed, your system likely does not support real-time prioritization.

```text
SCHED_OTHER min/max priority    : 0/0
SCHED_FIFO min/max priority     : 1/99
SCHED_RR min/max priority       : 1/99
SCHED_BATCH min/max priority    : 0/0
SCHED_IDLE min/max priority     : 0/0
SCHED_DEADLINE min/max priority : 0/
```

Now not all hope is lost, in case real-time priority is not supported you can very likely still give the node a high priority in the existing `SCHED_OTHER` policy by changing its nice value. The lowest possible niceness is -20 and this will ensure the highest priority among all the processes in the `SCHED_OTHER` policy. To set the niceness of a process, you can use the `nice` command in the launch-prefix.

### STEP 2: Give your user permission to set real-time priority without root privileges

Congratulations, you have a system that supports real-time priority! Now you need to give your user permission to set real-time priority without root privileges.
To do this, you need to edit the `/etc/security/limits.conf` file. Open it with your favorite text editor and add the following lines at the end of the file:

```text
# Allow user to set real-time priority
<username>   soft   rtprio     99
<username>   hard   rtprio     99
```

Then you must check that the following line is present in `/etc/pam.d/common-session`:

```text
session required pam_limits.so
```
If it is not present, add it to the end of the file. This line ensures that the limits set in `/etc/security/limits.conf` are applied to all sessions. Also check that it present in `/etc/pam.d/login`. If yes, you are good to go and reboot the system.

After logging in run the following command to check if the changes have been applied:

```bash
ulimit -r
```
The output should be `99` if the changes have been applied correctly. If it is `0`, you may need to log out and log back in or restart your system and ensure that the changes were made correctly in the files mentioned above. 

### STEP 3: Run the nodes with real-time priority
To run the nodes with real-time priority, you can use the `chrt` command in the launch-prefix. For example, to run a node with `SCHED_FIFO` policy and a priority of 99, you can use the following command:

```bash
chrt -f 99 roslaunch <package> <launchfile>
```

Furthermore, you can bind a node to a specific CPU core using the `taskset` command. For example, to bind a node to CPU core 0, you can use the following command:

```bash
taskset -c 0 chrt -f 99 roslaunch <package> <launchfile>
```

Running taskset does not require root privileges in any user. In case you are setting the niceness instead the steps to give yourself permission to allow setting the niceness without root privileges are quite similar, just use `nice` instead of `rtprio` in `limits.conf`. You can then use the `nice` command with taskset in the launch-prefix to achieve the same effect.

Important Note: Any `SCHED_FIFO` or `SCHED_RR` process, even with the lowest priority in these schedulers, will be higher in priority than the highest priority process within `SCHED_OTHER`. Running too many `SCHED_FIFO` or `SCHED_RR` processes can lead to starvation of other processes, so ideally don't run more than 2-3 such nodes, that too only if you have at least 5-6 CPU cores. 