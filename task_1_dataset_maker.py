import essentia.standard as es
import os 
import numpy as np
import json
import streamlit as st

class track_loader(object):
    def __init__(self, track_path: str):
        self.data_path = track_path
        self.features = {}
    
    def load_audio(self):
        # compute all the necessary audio signals for feature extraction:
        # y_stereo_44100 for loudness
        # y_mono_44100 for all others dsp models
        # y_mono_16000 for all ML models except TempoCNN
        # y_mono_11025 for tempo
        y_og, fs_og, n_channels, _, _, _ = es.AudioLoader(filename=self.data_path)()
        # y_stereo_44100
        if n_channels == 2 and fs_og == 44100:
            y_stereo_44100 = y_og
        elif n_channels == 2 and fs_og != 44100:
            y_mono_left  = y_og[:,0]
            y_mono_right = y_og[:,1]
            y_mono_left_44100  = es.Resample(inputSampleRate=float(fs_og), outputSampleRate=float(44100), quality=1).compute(y_mono_left)
            y_mono_right_44100 = es.Resample(inputSampleRate=float(fs_og), outputSampleRate=float(44100), quality=1).compute(y_mono_right)
            y_stereo_44100 = np.empty((np.size(y_mono_left_44100),2))
            y_stereo_44100[:,0] = y_mono_left_44100
            y_stereo_44100[:,1] = y_mono_right_44100
        elif n_channels == 1 and fs_og == 44100:
            y_stereo_44100 = np.empty((np.size(y_og),2))
            y_stereo_44100[:,0] = y_og
            y_stereo_44100[:,1] = y_og
        elif n_channels == 1 and fs_og != 44100:
            y_mono_44100 = es.Resample(inputSampleRate=float(fs_og), outputSampleRate=float(44100), quality=1).compute(y_og)
            y_stereo_44100 = np.empty((np.size(y_mono_44100),2))
            y_stereo_44100[:,0] = y_mono_44100
            y_stereo_44100[:,1] = y_mono_44100
        else:
            print("Signal that is neither mono nor stereo is being loaded. Not pretty code is being used for this case.")
            y_mono = np.mean(y_og, axis = 1)
            y_mono_44100 = es.Resample(inputSampleRate=float(fs_og), outputSampleRate=float(44100), quality=1).compute(y_mono) 
            y_stereo_44100 = np.empty((np.size(y_mono_44100),2))
            y_stereo_44100[:,0] = y_mono_44100
            y_stereo_44100[:,1] = y_mono_44100
        y_mono_44100 = es.MonoMixer()(y_stereo_44100, 2)
        y_mono_16000 = es.Resample(inputSampleRate=float(44100), outputSampleRate=float(16000), quality=1).compute(y_mono_44100)
        y_mono_11025 = es.Resample(inputSampleRate=float(44100), outputSampleRate=float(11025), quality=1).compute(y_mono_44100)
        return y_stereo_44100, y_mono_44100, y_mono_16000, y_mono_11025
    
    def __str__(self):
        return "track_loader: Track corresponding to the file located in "+self.data_path

class dataset_loader(object):
    def __init__(self, data_path: str):
        self.data_path = data_path
        tracks = {}
        allowed_extensions = {"mp3", "wav", "lac", "aac"}
        for root, dirs, files in os.walk(self.data_path):
            for file_name in files:
                if file_name[-3:] in allowed_extensions:
                    string_temporal = str(os.path.join(root, file_name))
                    tracks[file_name] = track_loader(string_temporal)
        self.tracks = tracks
        print("dataset_loader: "+str(len(tracks))+" tracks succesfully loaded :)")
    
    def __str__(self):
        return "dataset_loader: Dataset containing the files in the directory "+self.data_path
    
class feature_computer(object):
    def __init__(self):
        self.tempo_model              = es.TempoCNN(graphFilename="Resources/models/deepsquare-k16-3.pb")
        self.key_temperley_model      = es.KeyExtractor(profileType="temperley")
        self.key_krumhansl_model      = es.KeyExtractor(profileType="krumhansl")
        self.key_edma_model           = es.KeyExtractor(profileType="edma")
        self.loudness_model           = es.LoudnessEBUR128()
        self.embeddings_discogs_model = es.TensorflowPredictEffnetDiscogs(graphFilename="Resources/models/discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
        self.embeddings_MSD_model     = es.TensorflowPredictMusiCNN(graphFilename="Resources/models/msd-musicnn-1.pb", output="model/dense/BiasAdd")
        self.style_model              = es.TensorflowPredict2D(graphFilename="Resources/models/genre_discogs400-discogs-effnet-1.pb", input="serving_default_model_Placeholder", output="PartitionedCall:0")
        self.style_metadata           = json.load(open("Resources/models/discogs-effnet-bs64-1.json", 'r'))
        self.voice_instrumental_model = es.TensorflowPredict2D(graphFilename="Resources/models/voice_instrumental-discogs-effnet-1.pb",output="model/Softmax")
        self.danceability_model       = es.Danceability()
        self.arousal_valence_model    = es.TensorflowPredict2D(graphFilename="Resources/models/emomusic-msd-musicnn-2.pb", output="model/Identity")

    def tempo_extractor(self, audio: np.array):
        # input audio with fs = 11025
        tempo_global, _ , _ = self.tempo_model(audio)
        return tempo_global
    
    def key_temperly_extractor(self, audio: np.array):
        # input audio with fs = 44100
        return self.key_temperley_model(audio)

    def key_edma_extractor(self, audio: np.array):
        # input audio with fs = 44100
        return self.key_krumhansl_model(audio)

    def key_krumhansl_extractor(self, audio: np.array):
        # input audio with fs = 44100
        return self.key_edma_model(audio)
    
    def loudness_extractor(self, audio: np.array):
        # input audio with fs = 44100 and stereo
        _, _, integrated_loudness, _ = self.loudness_model(audio)
        return integrated_loudness

    def embeddings_discogs_extractor(self, audio: np.array):
        # input audio with fs = 16000
        return self.embeddings_discogs_model(audio)
    
    def embeddings_MSD_extractor(self, audio: np.array):
        # input audio with fs = 16000
        return self.embeddings_MSD_model(audio)

    def style_extractor(self, discogs_embeddings):
        # The input of this function is whatever is the result from the functoin embeddings_discogs_extractor
        classes = self.style_metadata["classes"]
        activations = np.mean(self.style_model(discogs_embeddings), axis=0)
        return dict(zip(classes, activations.tolist()))
    
    def voice_instrumental_extractor(self, discogs_embeddings):
        # The input of this function is whatever is the result from the functoin embeddings_discogs_extractor
        return tuple(np.mean(self.voice_instrumental_model(discogs_embeddings), axis=0))[0]

    def danceability_extractor(self, audio: np.array):
        # input audio with fs = 44100
        return self.danceability_model(audio)
    
    def arousal_valence_extractor(self, music_cnn_embeddings):
        # The input of this function is whatever is the result from the function embeddings_MSD_extractor
        return tuple(np.mean(self.arousal_valence_model(music_cnn_embeddings), axis=0).tolist())

def feature_extraction(dataset, bar, comment, GUI):    
    # all models are initiliazed
    models = feature_computer()
    total = len(dataset.tracks)
    # extraction starts
    iteration = 0
    for key, value in dataset.tracks.items():
        # progress bar 
        progress_percent = int(1000*iteration/total)/10
        if GUI == True:
            bar.progress(int(progress_percent))
            comment.text(str(progress_percent)+"%: "+str(iteration)+" out of "+str(total)+" completed.")
        # Compute all the versions of the signal
        y_stereo_44100, y_mono_44100, y_mono_16000, y_mono_11025 = value.load_audio()
        # Apply each feature extraction method to the audio file
        tempo_result              = models.tempo_extractor(y_mono_11025)
        key_temperly_result       = models.key_temperly_extractor(y_mono_44100)
        key_edma_result           = models.key_edma_extractor(y_mono_44100)
        key_krumhansl_result      = models.key_krumhansl_extractor(y_mono_44100)
        loudness_result           = models.loudness_extractor(y_stereo_44100)
        embeddings_discogs_result = models.embeddings_discogs_extractor(y_mono_16000)
        embeddings_MSD_result     = models.embeddings_MSD_extractor(y_mono_16000)
        style_result              = models.style_extractor(embeddings_discogs_result)
        voice_instrumental_result = models.voice_instrumental_extractor(embeddings_discogs_result)
        danceability_result       = models.danceability_extractor(y_mono_44100)
        arousal_valence_result    = models.arousal_valence_extractor(embeddings_MSD_result)
        # Assign computed features to the track
        dataset.tracks[key].features = {
            "tempo":              tempo_result,
            "key_temperly":       key_temperly_result,
            "key_edma":           key_edma_result,
            "key_krumhansl":      key_krumhansl_result,
            "loudness":           loudness_result,
            "embeddings_discogs": np.mean(embeddings_discogs_result, axis=0),
            "embeddings_MSD":     np.mean(embeddings_MSD_result,     axis=0),
            "style":              style_result,
            "voice_instrumental": voice_instrumental_result,
            "danceability":       danceability_result,
            "arousal_valence":    arousal_valence_result
        }
        iteration = iteration + 1
        print("iteration ",iteration, " done.")
        if GUI == True:
            bar.progress(100)
            comment.text("100%: "+str(iteration)+" out of "+str(total)+" completed.")

    return dataset

def dataset_to_numpy(dataset):
    # Initialize as a list
    numpy_result = []
    # Append the first row
    numpy_result.append(np.array([dataset.data_path, 0, 0], dtype=object))
    # Iterate through dataset.tracks
    for key, value in dataset.tracks.items():
        temporal_row = np.array([key, value.data_path, value.features], dtype=object)
        numpy_result.append(temporal_row)
    # Convert the list to a NumPy array
    numpy_result = np.vstack(numpy_result)
    return numpy_result

def numpy_to_dataset(dataset_npy):
    data_path = dataset_npy[0,0]
    dataset = dataset_loader(data_path=data_path)
    for key, value in dataset.tracks.items():
        row_aim = np.where(dataset_npy[:, 0] == key)[0]
        # dataset[key].data_path = dataset_npy[row_aim,1]
        dataset.tracks[key].features = dataset_npy[row_aim,2]
    return dataset