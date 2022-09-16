import os
import numpy as np
import scipy.ndimage # For loading and upsampling waveform data
import scipy.io # Load mat files
import uuid # Gives a "random" name
import h5py # Our storage format
import blosc # Compression

def ingest_ecg_worker(dat_files, destination):
    # Construct a "random" name for our archive.
    version_hash = uuid.uuid4().hex
    # The destination directory is the current directory
    # The output name for the HDF5 archive we write to disk
    output_name = 'BROAD_ml4h_klarqvist___physionet__waveforms__' + version_hash + '.h5'

    # With this new HDF5 file as `writef`. This syntax will automatically close the file handle
    # and flush when the loop ends.
    it = 1
    with h5py.File(os.path.join(destination,output_name), "w") as writef:
        # Loop over each row in our `dat_files` data frame
        # `i`` is an incremental counter, and `dat` is the `Series` (i.e. row) at offset `i``.
        for i, dat in dat_files.iterrows():
            # Periodically print progress
            if it % 100 == 0:
                print(f"{it}/{len(dat_files)+1}")
            
            # Load an ECG waveform from disk using the `loadmat` function in scipy. This
            # is simply how the organizers of the PhysioNet challenge decided to store them.
            local = np.array(scipy.io.loadmat(dat['mat'])['val'])
            
            # The St Peterburg dataset is sampled at 257 Hz, unlike the 500 Hz for most
            # of the other datasets. We will simply resample from 257 to 500 Hz. We will
            # "cheat" a bit and use the `zoom` functionality in scipy.
            if dat['sampling_frequency'] == 257:
                local = scipy.ndimage.zoom(local, (1, 500/257))
            # The PTB dataset is sampled at 1000 Hz which is twice that of the other
            # datasets. We can downsample to 500 Hz by selecting every second data point.
            elif dat['sampling_frequency'] == 1000:
                local = local.copy()[:,0:local.shape[1]:2] # Downsample to 500 Hz
                # Force C-order
                local = np.asarray(local, order='C')
            
            # Assert that we have the correct number of leads
            assert(local.shape[0] == 12)

            # Use some aggressive compression to store these waveforms. We can achieve
            # around 2.8 - 3-fold reduction in size, with a small decompression cost,
            # with these settings.
            compressed_data = blosc.compress(local, typesize=2, cname='zstd', clevel=9, shuffle=blosc.SHUFFLE)
            
            # Store the compressed data.
            dset = writef.create_dataset(dat.name, data=np.void(compressed_data))
            # Store the shape as meta data for the dataset
            dset.attrs['shape'] = local.shape
            # Increment counter
            it += 1

