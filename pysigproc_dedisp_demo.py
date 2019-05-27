#!/usr/bin/evn python

import numpy as np
from optparse import OptionParser
import math
from numba import cuda
import pylab as plt
from pysigproc import SigprocFile
import logging

logging_format = '%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)

def gpu_dmt(sf, dedisp_times, psr_data, max_delay, device=0):
    """
    :param cand: Candidate object
    :param device: GPU id
    :return:
    """
    cuda.select_device(device)
    dm_time = np.zeros((dedisp_times.shape[1], int(psr_data.shape[0]-max_delay)), dtype=np.float32)

    @cuda.jit(fastmath=True)
    def gpu_dmt(cand_data_in, all_delays, cand_data_out):
        ii, jj, kk = cuda.grid(3)
        if ii < cand_data_in.shape[0] and jj < cand_data_out.shape[1] and kk < all_delays.shape[1]:
            cuda.atomic.add(cand_data_out, (kk, jj), cand_data_in[ii, (jj + all_delays[ii,kk]) ]) 

    #with cuda.pinned(dedisp_times, dm_time, psr_data):
    all_delays = cuda.to_device(dedisp_times)
    dmt_return = cuda.device_array(dm_time.shape, dtype=np.float32)
    cand_data_in = cuda.to_device(np.array(psr_data.T, dtype=psr_data.dtype))

    threadsperblock = (4, 8, 32)
    blockspergrid_x = math.ceil(cand_data_in.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(cand_data_in.shape[1] / threadsperblock[1])
    blockspergrid_z = math.ceil(dm_list.shape[0] / threadsperblock[2])
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    gpu_dmt[blockspergrid, threadsperblock](cand_data_in, all_delays,  dmt_return)
    dm_time = dmt_return.copy_to_host()
    print(all_delays.shape)
    cuda.close()
    return dm_time


parser = OptionParser("")

parser.add_option("--filterbank", type="string", dest="filterbank", default="",
                  help="Filterbank file name.")
parser.add_option("--mindm", type="float", dest="mindm", default=0.0,
                  help="Minimum DM to dedisperse.")
parser.add_option("--maxdm", type="float", dest="maxdm", default=20000.000,
                  help="Maximum DM to dedisperse.")
parser.add_option("--dmstep", type="float", dest="dmstep", default=100.0,
                  help="DM step size.")
(options, args) = parser.parse_args()

gulp=1E12

sf = SigprocFile(fp="psr_p1s_dm10000.fil")

dm_list = np.arange(options.mindm, options.maxdm + options.dmstep, options.dmstep)

if (dm_list[-1] > options.maxdm):
    dm_list = np.delete(dm_list, -1)

dedisp_times = np.zeros((sf.nchans,len(dm_list)),dtype=np.int64)

for idx, dms in enumerate(dm_list):
    dedisp_times[:,idx] = np.round(-1*4148808.0*dms*(1 / (sf.chan_freqs[0])**2 - 1/(sf.chan_freqs)**2)/1000/sf.tsamp).astype('int32')

max_delay = dedisp_times.max() - dedisp_times.min()

if gulp < max_delay:
    logging.error('Gulp smaller than max dispersion delay')
    gulp = int(2*max_delay)

psr_data = sf.get_data(0,gulp)[:,0,:].copy()

dedisp_data = gpu_dmt(sf, dedisp_times, psr_data, max_delay, device=0)
print(dedisp_data.shape)
plt.imshow(dedisp_data,aspect='auto')
plt.show()
