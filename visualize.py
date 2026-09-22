import os
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import librosa.display
import chromagram as chrom
import numpy as np



def save_chroma(save_dir, save_name, audio, type="stft", hop_length=512, n_bins=12, **kwargs):
    """_summary_: save chromagram (x: chroma bin / bins, y: frame(time))

    Args:
        save_dir (str): path to the saving directory
        audio (AudioSegment object): in-memory PCM audio representation
        type (str, def="stft"): type of chromagram. Defaults to "stft".
        hop_length (int, def=512): shift distance per frame
        n_bins (int, def=12): number of bins for final chromagram
            n_choma for stft and cqt chroma, hpcp_size for hpcp. Defaults to 12.
        **kwargs:
            frame_size (int): number of samples in short block
            bins_per_octave (int): intermediate number of bins per octave | integer multiple of n_bins(n_chroma)
    """
    wav_amp, sample_rate = chrom.get_pcm_samples(audio)
    
    if type == "stft":
        chroma = chrom.get_stft_chroma(audio, n_fft=kwargs.get("frame_size", 2048), hop_length=hop_length, n_chroma=n_bins)
        title = "STFT Chroma"
    elif type == "cqt":
        chroma = chrom.get_cqt_chroma(audio, hop_length=hop_length, n_chroma=n_bins, bins_per_octave=kwargs.get("bins_per_octave", 12))
        # if bins_per_octave=36 and n_chroma=12 -> aggreated to 12 classes later
        title = "CQT Chroma"
    elif type == "hpcp":
        chroma = chrom.get_hpcp(audio, frame_size=kwargs.get("frame_size", 4096), hop_length=hop_length, hpcp_size=n_bins)
        title = "HPCP Chroma"
    
    plt.figure(figsize=(14, 5))

    if type=="hpcp":
        img = librosa.display.specshow(
            chroma,
            x_axis="time",
            y_axis="chroma",
            sr=sample_rate,
            hop_length=hop_length,
            norm=PowerNorm(0.5)
        )
        img.set_label("Chroma | Number of Bins")
    else:
        img = librosa.display.specshow(
            chroma,
            x_axis="time",
            y_axis="chroma",
            sr=sample_rate,
            hop_length=hop_length,
        )
    

    plt.colorbar(label="Strength") # shows color mapping
    plt.title(title)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True) # creates intermediate directory
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_spec(save_dir, save_name, audio, type="mel", hop_length=512, **kwargs):
    wav_amp, sample_rate = chrom.get_pcm_samples(audio)

    if type == "stft":
        spec = chrom.get_log_stft_spectrogram(audio, n_fft=kwargs.get("n_fft", 2048), hop_length=hop_length)
        title = "STFT Spectrogram"
        y_axis_name = "linear"
    elif type == "cqt":
        spec = chrom.get_log_cqt_spectrogram(audio, hop_length=hop_length, bins_per_octave=kwargs.get("bins_per_octave", 12))
        # if bins_per_octave=36 and n_chroma=12 -> aggreated to 12 classes later
        title = "CQT Spectrogram"
        y_axis_name = "cqt_hz"
    elif type == "mel":
        spec = chrom.get_log_mel_spectrogram(audio, n_fft=kwargs.get("n_fft", 4096), hop_length=hop_length)
        title = "MEL Spectrogram"
        y_axis_name = "mel"

    plt.figure(figsize=(14, 5))

    img = librosa.display.specshow(
        spec,
        x_axis="time",
        y_axis=y_axis_name,
        sr=sample_rate,
        hop_length=hop_length,
    )

    plt.colorbar(label="Strength") # shows color mapping
    plt.title(title)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True) # creates intermediate directory
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_onset_env(save_dir, save_name, onset_env, sample_rate, hop_length=512, title="Onset Envelope"):
    """_summary_

    Args:
        save_dir (_type_): _description_
        save_name (_type_): _description_
        onset_env (_type_): _description_
        sample_rate (_type_): _description_
        hop_length (int, optional): _description_. Defaults to 512.
        title (str, optional): _description_. Defaults to "Onset Envelope".
    """
    times = librosa.frames_to_time(
        np.arange(len(onset_env)),
        sr=sample_rate,
        hop_length=hop_length,
    )

    plt.figure(figsize=(14, 5))

    plt.plot(times, onset_env)
    plt.xlabel("Time (s)")
    plt.ylabel("Onset Strength")
    plt.title(title)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True) # creates intermediate directory
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_onset_beat(save_dir, save_name, onset_env, beat_times, sample_rate, hop_length=512, start_time=0, end_time=1000, title="Onset with Beat"):
    times = librosa.frames_to_time(
        np.arange(len(onset_env)),
        sr=sample_rate,
        hop_length=hop_length,
    )

    plt.figure(figsize=(14, 5))

    plt.plot(times, onset_env, label="Onset strength",)

    for beat_time in beat_times:
        if start_time <= beat_time <= end_time:
            plt.axvline(
                x=beat_time,
                linestyle="--",
                alpha=0.5,
            )

    plt.xlim(start_time, min(end_time, beat_times[-1])) # visible time range
    plt.xlabel("Time (s)")
    plt.ylabel("Onset Strength")
    plt.title(title)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True) # creates intermediate directory
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()