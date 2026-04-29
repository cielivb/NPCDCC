""" Simulating the top-level MPI workflow used for the big analysis """

import pandas as pd
import dask.dataframe as dd
import os

from time import sleep
from queue import PriorityQueue
from dask import delayed
from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
FILE_DIR = os.path.dirname(__file__)

ALL_TASKS_COMPLETE = False


@delayed
def read_feather(path):
    return pd.read_feather(path, use_threads=True)


if RANK == 0: # main rank
    
    # Load in dask dataframe from feather file
    feather_path = os.path.join(FILE_DIR, "data", "proofread_connections_783.feather")
    ddf = dd.from_delayed(read_feather(feather_path))
    
    # Get neuropil priorities via number of edges (lower number = higher priority)
    neuropils_counts_series = ddf['neuropil'].value_counts().compute()
    neuropil_priorities = list(neuropils_counts_series.map(lambda x: -x).items())
    
    # Allocate neuropils to worker processes on demand.
    # Use priority queue object to reduce chance of one worker process working
    # significantly longer than the other worker processes towards the end of
    # the queue.
    neuropil_queue = PriorityQueue()
    for item in neuropil_priorities:
        neuropil_queue.put(item)
    
    # Index represents worker state (0 = idle, 1 = active).
    # Index 0 (main rank) remains at 0.
    worker_state = [0 for _ in range(COMM.Get_size())]
    
    status = MPI.Status()
    while not neuropil_queue.empty:
        # Get ranks of workers that are currently idle
        idle = [rank for rank, state in enumerate(worker_state) if state == 0]
        
        # Allocate tasks to idle workers
        for rank in idle:
            COMM.send(neuropil_queue.pop(), rank)
            worker_state[rank] == 1
        
        sleep(5)
        
        # Receive worker task complete messages and update worker states
        while COMM.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status):
            msg = COMM.recv(source=status.Get_source(), tag=status.Get_tag())
            idle_rank = status.Get_source()
            worker_state[idle_rank] == 0
    
    # Send termination signal
    ALL_TASKS_COMPLETE = True
    COMM.Ibcast(ALL_TASKS_COMPLETE, root=0)
    
else: # Worker rank
    ALL_TASKS_COMPLETE = COMM.Ibcast(ALL_TASKS_COMPLETE, root=0)
    while not ALL_TASKS_COMPLETE:
        
        # Receive incoming message
        msg = None
        if COMM.Iprobe(source=0, tag=MPI.ANY_TAG, status=None):
            msg = COMM.recv(source=0, tag=MPI.ANY_TAG)
        
        if msg:
            print(msg)
            sleep(2)
            COMM.send("Task complete", 0)
        else:
            sleep(5)

COMM.Barrier()