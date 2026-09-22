import numpy as np
import librosa
import essentia.standard as es


def get_pcm_samples(audio):
    """_summary_: computes wave_amplitude function and sample rate

    Args:
        audio (AudioSegment object): in-memory PCM audio representation
    Returns:
        pcm_float (float ndarray): x: sample n | y: amplitude (in PCM float in range [-1.0, 1.0)] )
        sample rate (int): n samples / second
    """

    if audio.channels == 1: # mono
        audio.set_channels(1)
    
    samples = np.asarray(audio.get_array_of_samples()) # raw audio -> numeric samples, x: sample n | y: audio sample
    bits_sample = audio.sample_width * 8 # 16
    pcm_float = samples.astype(np.float32) / (2**(bits_sample - 1)) # signed sample-bit PCM(pulse code modulation) / precision: 32 bit floating point
    sample_rate = audio.frame_rate # sample rate: n samples / second

    return pcm_float, sample_rate


def get_croped_chroma(audio, chroma, frame_length, hop_length, threshold=-60.0):
    """_summary_: cuts start and end chroma is the db is lower than the shreshold

    Args:
        audio (pydub.AudioSegment object): in-memory PCM audio representation
        chroma (np.ndarray): (n_chroma, n_frames) size chromagram
        frame_length (int): RMS analysis frame length in samples (rms_frame_length for CQT)
        hop_length (int): shift distance per frame
        threshold (float, def=-60.0): silence threshold relative to maximum RMS, in dB
    Returns:
        chroma (np.ndarray): cropped chromagram
    """
    wav_amp, sample_rate = get_pcm_samples(audio)
    rms = librosa.feature.rms(y=wav_amp, frame_length=frame_length, hop_length=hop_length, center=False)[0] # root mean square
    rms_db = librosa.amplitude_to_db(rms, ref=np.max,)
    act_frames = min(chroma.shape[1], len(rms_db)) # exact end

    chroma = chroma[:, :act_frames].copy()
    rms_db = rms_db[:act_frames]

    active = rms_db >= threshold
    if np.any(active):
        start = np.argmax(active)
        end = len(active) - np.argmax(active[::-1])
        chroma = chroma[:, start:end]

    return chroma


def get_stft_chroma(audio, n_fft=2048, hop_length=512, n_chroma=12, crop=True):
    """_summary_: waveform > stft > magnitude/power spectrum > freq to pitch classes > 12 x Time chromagram

    Args:
        audio (AudioSegment object): in-memory PCM audio representation
        n_fft (int): number of samples per FFT
        hop_length (int): shift distance per frame
        n_chroma (int): number of pitch class bins
    Returns:
        stft_chroma (12, n_frames): x: Fourier frequency in 12 pitch class bins | y: time in num_frames
    """

    wav_amp, sample_rate = get_pcm_samples(audio)

    stft_chroma = librosa.feature.chroma_stft(
        y=wav_amp, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_chroma=n_chroma, center=False
    )
    if crop:
        crop_stft_chroma = get_croped_chroma(audio, stft_chroma, n_fft, hop_length)
        return crop_stft_chroma # stft_chroma
    else:
        return stft_chroma


def get_cqt_chroma(audio, hop_length=512, n_chroma=12, bins_per_octave=12, n_octaves=7):
    """_summary_: waveform > cqt > magnitude/power spectrum > freq to pitch classes (CQT frequency in 12 bins) > 12 x Time chromagram

    Args:
        audio (AudioSegment object): in-memory PCM audio representation
        hop_length (int): shift distance per frame
        n_chroma (int): number of pitch class bins
        bins_per_octave (int): how many frequency bins per octave
            e.g. 12 or 36
        n_octaves: range of octave coverage
    Returns:
        cqt_chroma (12, n_frames): x: CQT frequency in 12 pitch class bins | y: time in num_frames
            Constant-Q transform (CQT) converts frequency in logarithmic
            semitone has constant ratio (e.g. A4 = 440Hz, A5 = 880Hz)
            follow logarithmic, pitch-oriented spacing
    """

    wav_amp, sample_rate = get_pcm_samples(audio)

    cqt_chroma = librosa.feature.chroma_cqt(
        y=wav_amp, sr=sample_rate, hop_length=hop_length, n_chroma=n_chroma, bins_per_octave=bins_per_octave, n_octaves=n_octaves
    )

    rms_frame_length = 2048 # convention
    crop_cqt_chroma = get_croped_chroma(audio, cqt_chroma, rms_frame_length, hop_length)

    return crop_cqt_chroma # cqt_chroma


def get_hpcp(audio, frame_size=4096, hop_length=512, hpcp_size=12, minFreq=40, maxFreq=5000, crop=True):
    """_summary_: waveform > FFT > magnitude/power spectrum > spectral peak detection > frequencies + magnitudes > HPCP weighting > HPCP

    Args:
        audio (AudioSegment object): in-memory PCM audio representation
        frame_size (int, def=4096): frame window size
        hop_length (int, def=512): shift distance per frame
        hpcp_size (int, def=12): number of bin for hpcp
    Returns:
        cqt_chroma (12, n_frames): x: CQT frequency in 12 pitch class bins | y: time in num_frames
            Constant-Q transform (CQT) converts frequency in logarithmic
            semitone has constant ratio (e.g. A4 = 440Hz, A5 = 880Hz)
            follow logarithmic, pitch-oriented spacing
    """

    wav_amp, sample_rate = get_pcm_samples(audio)

    window = es.Windowing(type="blackmanharris62") # smooth weighting function object
    spectrum = es.Spectrum(size=frame_size) # FFT transform (with abs) object

    spectral_peaks = es.SpectralPeaks(
        sampleRate=sample_rate,
        orderBy="magnitude",
        magnitudeThreshold=0,
        minFrequency=minFreq,
        maxFrequency=min(maxFreq, sample_rate / 2.0),
        maxPeaks=60,
    ) # later mkae this i 
    # magnitudeThreshold=1e-5

    # print("wav:", wav_amp.dtype, wav_amp.min(), wav_amp.max())
    # print("sample rate:", sample_rate)

    hpcp_algorithm = es.HPCP(
        size=hpcp_size, sampleRate=sample_rate,
        minFrequency=minFreq, maxFrequency=min(maxFreq, sample_rate / 2.0), referenceFrequency=440.0,
        weightType="squaredCosine",
        windowSize=4.0/3.0,
        harmonics=8,
        normalized="unitMax",
    ) # minFrequency .. is the baseline reference setting
    hpcp_frames = []

    for frame in es.FrameGenerator(wav_amp, frameSize=frame_size, hopSize=hop_length, startFromZero=True, lastFrameToEndOfFile=False):
        windowed = window(frame)
        spec = spectrum(windowed)
        freq, mag = spectral_peaks(spec)
        hpcp = hpcp_algorithm(freq, mag,)
        hpcp_frames.append(hpcp)

    hpcp = np.asarray(hpcp_frames, dtype=np.float32).T
    if crop:
        crop_hpcp = get_croped_chroma(audio, hpcp, frame_length=frame_size, hop_length=hop_length)

        return crop_hpcp
    else:
        return hpcp

# def chord_seq():
#     # Implement Traditional Method later on another file
#     return 0

# --------------- spectrograms --------------- #
def get_log_stft_spectrogram(audio, n_fft=2048, hop_length=512):
    """_summary_: waveform > stft > magnitude/power spectrum

    Args:
        audio (AudioSegment object): in-memory PCM audio representation
        n_fft (int): number of samples per FFT
        hop_length (int): shift distance per frame
    Returns:
        stft_spectrogram (samp_rate/hop_length, n_frames): x: Fourier frequency (in db) | y: time in num_frames
        20 * log_10 (A/A_ref), reference amplitude value is the maximum value on the window
    """
    wav_amp, samp_rate = get_pcm_samples(audio)

    stft_spec = librosa.stft(wav_amp, n_fft=n_fft, hop_length=hop_length, center=False)
    stft_db = librosa.amplitude_to_db(np.abs(stft_spec), ref=np.max,)

    return stft_db


def get_log_cqt_spectrogram(audio, hop_length=512, bins_per_octave=12):
    """_summary_: waveform > cqt > magnitude/power spectrum
        f_k = f_min 2^(k/B) (B = bins per octave) < bin center spacing
    
    Args:
        audio (AudioSegment object): in-memory PCM audio representation
        hop_length (int): shift distance per frame
        bins_per_octave (int): how many frequency bins per octave
    Returns:
        cqt_chroma (samp_rate/hop_length, n_frames): x: CQT frequency | y: time in num_frames
        20 * log_10 (A/A_ref), reference amplitude value is the maximum value on the window
    """
    wav_amp, samp_rate = get_pcm_samples(audio)

    cqt_spec = librosa.cqt(wav_amp, hop_length=hop_length, bins_per_octave=bins_per_octave)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt_spec), ref=np.max,)

    return cqt_db


def get_log_mel_spectrogram(audio, n_fft=2048, hop_length=512, power=2.0):
    """_summary_: waveform > stft > magnitude/power spectrum > Mel filter bank
        Mel-filter boundary
            HTK formula m(f) = 2595 log_10 (1+f/700)
            librosa uses Slaney-style area normalization (linear below 1kHz, logarithmic above 1kHz)
        Weighting funciton of m-th mel filter > triangle

    Args:
        audio (AudioSegment object): in-memory PCM audio representation
        n_fft (int): number of samples per FFT
        hop_length (int): shift distance per frame
    Returns:
        mel_spectrogram (samp_rate/hop_length, n_frames): x: mel frequency | y: time in num_frames
        10 * log_10 (P/P_ref), reference power value is the maximum value on the window
    """
    wav_amp, samp_rate = get_pcm_samples(audio)

    mel_spec = librosa.feature.melspectrogram(y=wav_amp, sr=samp_rate, n_fft=n_fft, hop_length=hop_length, power=power)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max,) # power = amp^2

    return mel_db


def get_beat_chroma(chroma, beat_frames):
    """_summary_: produces beat-chroma (for each beat, average chroma value)

    Args:
        chroma (samp_rate/hop_length, n_frames): x: frequency | y: time in num_frames
        beat_frames (num_beat_frames,): each number indicate the frame number of chromagram
    Returns:
        beat_chroma (12, num_beat_frames-1): x: beat intervals | y: 12 chroma bins
            value: average chroma value within each beat interval
    """
    beat_chroma = []

    for i in range(len(beat_frames)-1): # start position looping
        sta = max(beat_frames[i], 0)
        end = min(beat_frames[i+1], chroma.shape[1]) # bound between 0 and chroma width(x-axis)
        
        if end <= sta:
            continue

        avg_chroma = np.mean(chroma[:, sta:end], axis=1)
        beat_chroma.append(avg_chroma) # (beat_frames, 12)

    beat_chroma = np.asarray(beat_chroma, dtype=np.float32) # ndarray

    norms = np.linalg.norm(beat_chroma, axis=1, keepdims=True)
    beat_chroma = beat_chroma / np.maximum(norms, 1e-10)

    beat_chroma = beat_chroma.T # (12, beat_frames)

    return beat_chroma


def get_beat_hpcp(hpcp, beat_frames):
    """
    Args:
        hpcp: shape (n_hpcp_bins, n_frames)
              rows: HPCP bins
              columns: time frames

        beat_frames: shape (n_beats,)
                     beat positions expressed in frame indices

    Returns:
        beat_hpcp: shape (n_hpcp_bins, n_beat_intervals)
                   each column is the average HPCP vector
                   between two consecutive beats
    """

    beat_hpcp = []

    for i in range(len(beat_frames) - 1):
        sta = max(beat_frames[i], 0)
        end = min(beat_frames[i+1], hpcp.shape[1])

        if end <= sta:
            continue

        avg_hpcp = np.mean(hpcp[:, sta:end], axis=1)
        beat_hpcp.append(avg_hpcp) # (beat_frames, 12)

    beat_hpcp = np.asarray(beat_hpcp, dtype=np.float32) # ndarray

    norms = np.linalg.norm(beat_hpcp, axis=1, keepdims=True) # L2 normalization
    beat_hpcp = beat_hpcp / np.maximum(norms, 1e-10)

    beat_hpcp = beat_hpcp.T # (12, beat_frames)

    return beat_hpcp