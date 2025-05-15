Log into the HPC system  
1. putty  
   username: wtang@cssl-master.soc.cuhk.edu.hk  
   password: clayton  
   Like the linux, you should give instruction here.  
3. WinSCP
   zonename: cssl-master.soc.cuhk.edu.hk
   username: wtang
   password: ...
   You should store your code and data here, just like the autodl.  
   Before log out, you must save your working zone, otherwise, the files will be gone.  


Some basic instructions to use the putty (Which is Linux Based).
1. cat 
wtang@cssl-master:~$ cat sample.script  
This is to show the sample.script file. By this, you can see a specific file in the WinSCP    

2. history  
wtang@cssl-master:~$ history  
This is to show the record of all the past instructions

3. squeue
wtang@cssl-master:~$ squeue  
This is to show the current users and tasks in the multi-thread computing systems

4. ls -lat
wtang@cssl-master:~$ ls -lat
This is to show the record of all the files you've uploaded to the WinSCP or HPC has output to the WinSCP

5. exit
wtang@cssl-master:~$ exit  
To log out or exit from the GPU currently connected to.

6. pwd  
wtang@cssl-master:~$ pwd
To show the current working folder
   wtang@cssl-master:~$ pwd  
   /home/wtang  
   wtang@cssl-master:~$ cd /home/wtang/Llama3/  
   wtang@cssl-master:~/Llama3$ pwd  
   /home/wtang/Llama3  
   wtang@cssl-master:~/Llama3$ cd ..  
   wtang@cssl-master:~$ pwd  
   /home/wtang
   
8. ssh  
To get connected to the GPU03, but why shall I do this???

   wtang@cssl-master:~$ ssh gpu03
   wtang@gpu03's password:
   Welcome to Ubuntu 22.04.5 LTS (GNU/Linux 5.15.0-131-generic x86_64)
   
    * Documentation:  https://help.ubuntu.com
    * Management:     https://landscape.canonical.com
    * Support:        https://ubuntu.com/pro
   
   This system has been minimized by removing packages and content that are
   not required on a system that users do not log into.
   
   To restore this content, you can run the 'unminimize' command.
   New release '24.04.1 LTS' available.
   Run 'do-release-upgrade' to upgrade to it.
   
   Last login: Tue Feb 18 17:47:05 2025 from 137.189.167.201
   
   wtang@gpu03:~$ exit
   logout
   Connection to gpu03 closed.

Some more important instructions:  
https://blog.csdn.net/LeviLizhi/article/details/142451228  

sinfo  
# to know the idle GPU Server
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST  
rtx4090*     up   infinite      1    mix gpu04  
rtx4090*     up   infinite      1   idle gpu03  

sbatch test.abc  
to submit a task  

scancel 159  
to cancel a task  

ssh gpu03  
to get connected to the GPU server 03  
then you should input  

nvidia-smi  
this shows the usage of the GPU  
+-----------------------------------------------------------------------------------------+  
| Processes:                                                                              |  
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |  
|        ID   ID                                                               Usage      |  
|=========================================================================================|  
|    0   N/A  N/A     68105      C   python3                                      4748MiB |  
|    1   N/A  N/A     68105      C   python3                                      5846MiB |  
|    2   N/A  N/A     68105      C   python3                                      6008MiB |  
