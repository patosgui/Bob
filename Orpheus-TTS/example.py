import numpy as np
from orpheus_cpp import OrpheusCpp
from scipy.io.wavfile import write

orpheus = OrpheusCpp(verbose=False)

text = "I really hope the project deadline doesn't get moved up again."
buffer = []
for i, (sr, chunk) in enumerate(
    orpheus.stream_tts_sync(text, options={"voice_id": "tara"})
):
    buffer.append(chunk)
    print(type(chunk))
    print(f"Generated chunk {i}")
print(buffer)
buffer = np.concatenate(buffer, axis=1)
# Check data type and range
print(f"Data type: {buffer.dtype}")
print(f"Min value: {np.min(buffer)}")
print(f"Max value: {np.max(buffer)}")
print(buffer[0])
print(buffer[0].shape)
print(buffer[0][0].size)
write("output.wav", 24000, buffer[0])
# write("output.wav", 24_000, buffer)
