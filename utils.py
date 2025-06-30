import numpy as np
import mne
from scipy import signal
import matplotlib.pyplot as plt

def load_muse_csv(filename, sfreq=256.0, ch_names=None):
    """Load Muse S EEG CSV data"""
    raw_data = np.loadtxt(filename, delimiter=',', skiprows=1)
    if ch_names is None:
        ch_names = ['TP9', 'AF7', 'AF8', 'TP10']

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(raw_data.T, info)
    raw.set_montage('standard_1020', match_case=False)
    return raw

def load_fif_file(filename, sfreq=250, ch_names=None):
    return mne.io.read_raw(filename, preload=True)

def bandpass_filter(raw, l_freq=1.0, h_freq=40.0):
    """Bandpass filter EEG data"""
    return raw.copy().filter(l_freq, h_freq)

def detect_artifacts(raw, threshold=100e-6, min_duration=0.5):
    """Mark bad segments if signal exceeds threshold
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw data
    threshold : float
        Threshold for artifact detection in volts
    min_duration : float
        Minimum duration of artifact in seconds
    """
    annotations = []
    data, times = raw.get_data(return_times=True)
    sfreq = raw.info['sfreq']
    min_samples = int(min_duration * sfreq)  # Convert duration to samples
    
    for i, ch in enumerate(data):
        above_thresh = np.abs(ch) > threshold
        starts = np.where(np.diff(above_thresh.astype(int)) == 1)[0]
        ends = np.where(np.diff(above_thresh.astype(int)) == -1)[0]
        
        if len(starts) == 0 or len(ends) == 0:
            continue
            
        # Ensure starts come before ends
        if ends[0] < starts[0]:
            ends = ends[1:]
        if len(starts) > len(ends):
            starts = starts[:-1]
            
        # Combine nearby artifacts and filter out short ones
        final_starts = []
        final_ends = []
        
        current_start = starts[0]
        current_end = ends[0]
        
        for i in range(1, len(starts)):
            # If next artifact starts soon after current one ends
            if starts[i] - current_end < min_samples:
                current_end = ends[i]  # Extend current artifact
            else:
                # If current artifact is long enough, keep it
                if current_end - current_start >= min_samples:
                    final_starts.append(current_start)
                    final_ends.append(current_end)
                current_start = starts[i]
                current_end = ends[i]
        
        # Add last artifact if it's long enough
        if current_end - current_start >= min_samples:
            final_starts.append(current_start)
            final_ends.append(current_end)
        
        # Create annotations for the filtered artifacts
        for start, end in zip(final_starts, final_ends):
            onset = times[start]
            duration = times[end] - times[start]
            annotations.append(mne.Annotations(onset, duration, 'bad_artifact'))
    
    raw.set_annotations(sum(annotations, mne.Annotations([], [], [])))
    return raw

def run_ica(raw, n_components=None):
    """Run ICA to remove ocular/muscle artifacts"""
    n_channels = len(raw.ch_names)
    if n_components is None or n_components > n_channels:
        n_components = n_channels  # ICA cannot have more components than channels

    ica = mne.preprocessing.ICA(n_components=n_components, random_state=42)
    ica.fit(raw)
    # Find EOG artifacts (if EOG channel exists, otherwise skip)
    try:
        eog_inds, scores = ica.find_bads_eog(raw)
        ica.exclude = eog_inds
    except Exception:
        pass  # No EOG channel, skip

    raw_corrected = raw.copy()
    ica.apply(raw_corrected)
    return raw_corrected

def plot_psd(raw, fmin=1.0, fmax=40.0):
    """Plot power spectral density
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw data
    fmin : float
        Minimum frequency to include
    fmax : float
        Maximum frequency to include
    """
    # Get data length and sampling frequency
    n_times = raw.n_times
    sfreq = raw.info['sfreq']
    
    # Calculate appropriate segment length (use 2 seconds of data or less if not available)
    n_per_seg = min(int(2 * sfreq), n_times)
    
    # Ensure n_per_seg is even
    n_per_seg = n_per_seg if n_per_seg % 2 == 0 else n_per_seg - 1
    
    print(f"[*] Computing PSD with {n_per_seg} samples per segment")
    raw.compute_psd(fmin=fmin, fmax=fmax, n_per_seg=n_per_seg).plot()
    plt.show(block=True)  # This will keep the plot window open

def preprocess_pipeline(filename, sfreq=256.0):
    print("[*] Loading data...")
    # raw = load_muse_csv(filename, sfreq)
    raw = load_fif_file(filename)

    print("[*] Filtering...")
    raw = bandpass_filter(raw)

    print("[*] Detecting artifacts...")
    raw = detect_artifacts(raw)

    print("[*] ICA artifact correction...")
    raw = run_ica(raw)

    print("[*] Plotting PSD...")
    plot_psd(raw)

    return raw
