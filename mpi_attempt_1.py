import pandas as pd
import dask.dataframe as dd
import os

from queue import PriorityQueue
from dask import delayed
from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
FILE_DIR = os.path.dirname(__file__)



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
    print(neuropil_priorities)
    
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
        # For each idle worker:
        #    Pop item from priority queue
        #    Send to worker for processing
        # Sleep for an appropriate timeframe
        # Check to see if busy workers are still busy and update if necessary
        # while iprobe(any incoming message):
        #    receive message
        #    update worker_state of rank that sent message
    
    # Send termination signal 
    
else: # Worker rank
    all_tasks_complete = False
    while not all_tasks_complete:
        # Receive incoming message
        
        # If message exists and is task:
            # Sleep (mimic doing task)
            # Send finished message
        
        # Else if message exists and is all tasks complete
        #    all_tasks_complete = True
        
        # Else
        #    Sleep for appropriate time-span
        pass
    

