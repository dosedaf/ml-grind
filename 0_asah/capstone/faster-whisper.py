from faster_whisper import WhisperModel

model_size = "large-v3"

# Run on GPU with FP16
# model = WhisperModel(model_size, device="cpu", compute_type="float16")

# or run on GPU with INT8
model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe("tes.mp3", beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

s = ''

for segment in segments:
    s += segment.text

print(s)
