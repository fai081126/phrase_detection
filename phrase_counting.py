#!/usr/bin/env python
# coding: utf-8

# In[2]:


#pip install librosa


# In[3]:


#pip install numpy==1.23.5


# In[1]:


import librosa
import numpy as np
import matplotlib.pyplot as plt


# In[238]:


# Load the audio file
y, sr = librosa.load('ptk_2.wav')


# In[10]:


plt.figure()
librosa.display.waveshow(y, sr=sr)
plt.title('nutcracker waveform')
plt.show()


# In[241]:


o_env = librosa.onset.onset_strength(y=y, sr=sr)

times = librosa.times_like(o_env, sr=sr)


onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)


# In[242]:


D = np.abs(librosa.stft(y))

fig, ax = plt.subplots(nrows=2, sharex=True)

librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),

                         x_axis='time', y_axis='log', ax=ax[0])

ax[0].set(title='Power spectrogram')

ax[0].label_outer()

ax[1].plot(times, o_env, label='Onset strength')

ax[1].vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9,

           linestyle='--', label='Onsets')

ax[1].legend()


# In[249]:


time = times[onset_frames]
time = time[time>1]#time before 1.5 shall be no sound
count = len(time)/3
print("Approzimate Number of 'Pa Ta Ka' units:", int(count))

