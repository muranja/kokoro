# kokoro

An inference library for [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M). You can [`pip install kokoro`](https://pypi.org/project/kokoro/).

> **Kokoro** is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, Kokoro can be deployed anywhere from production environments to personal projects.

### Usage
You can run this cell on [Google Colab](https://colab.research.google.com/). [Listen to samples](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/SAMPLES.md).
```py
# 1️⃣ Install kokoro and soundfile
!pip install -q kokoro>=0.3.4 soundfile

# 2️⃣ Install espeak (used for English OOD fallback and some non-English languages)
!apt-get -qq -y install espeak-ng > /dev/null 2>&1

# 3️⃣ Import required libraries and initialize the pipeline
from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import numpy as np

# Choose a language and voice.
# For American English, use 'a'; for British English, 'b', etc.
pipeline = KPipeline(lang_code='a')  # ensure lang_code matches the chosen voice

# This text is for demonstration purposes only
text = '''
The sky above the port was the color of television, tuned to a dead channel.
"It's not like I'm using," Case heard someone say, as he shouldered his way through the crowd around the door of the Chat. "It's like my body's developed this massive drug deficiency."
It was a Sprawl voice and a Sprawl joke. The Chatsubo was a bar for professional expatriates; you could drink there for a week and never hear two words in Japanese.

These were to have an enormous impact, not only because they were associated with Constantine, but also because, as in so many other areas, the decisions taken by Constantine (or in his name) were to have great significance for centuries to come. One of the main issues was the shape that Christian churches were to take, since there was not, apparently, a tradition of monumental church buildings when Constantine decided to help the Christian church build a series of truly spectacular structures. The main form that these churches took was that of the basilica, a multipurpose rectangular structure, based ultimately on the earlier Greek stoa, which could be found in most of the great cities of the empire. Christianity, unlike classical polytheism, needed a large interior space for the celebration of its religious services, and the basilica aptly filled that need. We naturally do not know the degree to which the emperor was involved in the design of new churches, but it is tempting to connect this with the secular basilica that Constantine completed in the Roman forum (the so-called Basilica of Maxentius) and the one he probably built in Trier, in connection with his residence in the city at a time when he was still caesar.

[Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, [Kokoro](/kˈOkəɹO/) can be deployed anywhere from production environments to personal projects.
'''

# 4️⃣ Generate audio segments using the pipeline.
# The split_pattern here splits on one or more newlines.
generator = pipeline(
    text, 
    voice='af_heart',  # change voice here as needed
    speed=1, 
    split_pattern=r'\n+'
)

# Initialize a list to collect each generated audio segment.
audios = []

print("Generating individual audio segments...")
for i, (gs, ps, audio) in enumerate(generator):
    print(f"\nSegment {i}:")
    print("Graphemes:", gs)
    print("Phonemes:", ps)
    
    # Display each individual audio segment.
    display(Audio(data=audio, rate=24000, autoplay=(i == 0)))
    
    # Ensure audio is a NumPy array.
    audio_arr = np.array(audio)
    print(f"Segment {i} length (samples): {audio_arr.shape[0]}")
    
    audios.append(audio_arr)

# Check that we have at least one segment.
if not audios:
    raise ValueError("No audio segments were generated.")

# 5️⃣ Concatenate all audio segments into a single array.
# We're concatenating along the first axis (samples).
combined_audio = np.concatenate(audios, axis=0)
print(f"\nCombined audio length (samples): {combined_audio.shape[0]}")

# Display the combined audio for playback.
display(Audio(data=combined_audio, rate=24000, autoplay=True))

# 6️⃣ Save the combined audio into one file.
sf.write('combined.wav', combined_audio, 24000)
print("Combined audio saved as 'combined.wav'")

```

Under the hood, `kokoro` uses [`misaki`](https://pypi.org/project/misaki/), a G2P library at https://github.com/hexgrad/misaki

### Conda Environment

Use the following conda `environment.yml` if you're facing any dependency issues.
```yaml
name: kokoro
channels:
  - defaults
dependencies:
  - python==3.9       
  - libstdcxx~=12.4.0 # Needed to load espeak correctly. Try removing this if you're facing issues with Espeak fallback. 
  - pip:
      - kokoro>=0.3.1
      - soundfile
      - misaki[en]
```

### Acknowledgements

- 🛠️ [@yl4579](https://huggingface.co/yl4579) for architecting StyleTTS 2.
- 🏆 [@Pendrokar](https://huggingface.co/Pendrokar) for adding Kokoro as a contender in the TTS Spaces Arena.
- 📊 Thank you to everyone who contributed synthetic training data.
- ❤️ Special thanks to all compute sponsors.
- 👾 Discord server: https://discord.gg/QuGxSWBfQy
- 🪽 Kokoro is a Japanese word that translates to "heart" or "spirit". Kokoro is also the name of an [AI in the Terminator franchise](https://terminator.fandom.com/wiki/Kokoro).

<img src="https://static0.gamerantimages.com/wordpress/wp-content/uploads/2024/08/terminator-zero-41-1.jpg" width="400" alt="kokoro" />
