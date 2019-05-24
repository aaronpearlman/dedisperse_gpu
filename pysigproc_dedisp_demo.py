import sys
sys.path.append("/home/pearlman/software/fetch/pysigproc")

import subprocess
import numpy as np
import h5py
from optparse import OptionParser

from gpu_utils import gpu_dmt
import math
from numba import cuda

from pysigproc import SigprocFile

def gpu_dmt(sf, psr_data, mindm, maxdm, dmstep, device=0):
    """
    :param cand: Candidate object
    :param device: GPU id
    :return:
    """
    cuda.select_device(device)
    chan_freqs = cuda.to_device(np.array(sf.chan_freqs, dtype=np.float32))
    
    dm_list = np.arange(mindm, maxdm + dmstep, dmstep)
    
    if (dm_list[-1] > maxdm):
        dm_list = np.delete(dm_list, -1)
    
    dm_list = cuda.to_device(dm_list.astype(np.float32))
    dmt_return = cuda.to_device(np.zeros((np.shape(psr_data)[-1], np.shape(psr_data)[0]), dtype=np.float32))
    cand_data_in = cuda.to_device(np.array(psr_data.T, dtype=np.float32))

    @cuda.jit
    def gpu_dmt(cand_data_in, chan_freqs, dms, cand_data_out, tsamp):
        ii, jj, kk = cuda.grid(3)
        if ii < cand_data_in.shape[0] and jj < cand_data_in.shape[1] and kk < dms.shape[0]:
            disp_time = int(
                -1 * 4148808.0 * dms[kk] * (1 / (chan_freqs[0]) ** 2 - 1 / (chan_freqs[ii]) ** 2) / 1000 / tsamp)
            cuda.atomic.add(cand_data_out, (kk, jj), cand_data_in[ii, (jj + disp_time)])

    threadsperblock = (4, 8, 32)
    blockspergrid_x = math.ceil(cand_data_in.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(cand_data_in.shape[1] / threadsperblock[1])
    blockspergrid_z = math.ceil(dm_list.shape[0] / threadsperblock[2])

    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    gpu_dmt[blockspergrid, threadsperblock](cand_data_in, chan_freqs, dm_list, dmt_return, float(sf.tsamp))

    cuda.close()

    return dmt_return;



parser = OptionParser("")

parser.add_option("--filterbank", type="string", dest="filterbank", default="",
                  help="Filterbank file name.")
parser.add_option("--mindm", type="float", dest="mindm", default=0.0,
                  help="Filterbank file name.")
parser.add_option("--maxdm", type="float", dest="maxdm", default=0.0,
                  help="Filterbank file name.")
parser.add_option("--dmstep", type="float", dest="dmstep", default=0.0,
                  help="Filterbank file name.")
(options, args) = parser.parse_args()








sf = SigprocFile(fp="psr_p1s_dm10000.fil")

maxChunkSize = 1E12

psr_data = sf.get_data(0, maxChunkSize)
#print(sf.nsamples)
sf.nsamples = np.shape(psr_data)[0]



dedisp_data = gpu_dmt(sf, psr_data, options.mindm, options.maxdm, options.dmstep, device=0)
