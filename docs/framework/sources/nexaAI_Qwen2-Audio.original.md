Qwen2-Audio
We're bringing Qwen2-Audio to run locally on edge devices with Nexa-SDK, offering various GGUF quantization options.
Qwen2-Audio is a SOTA small-scale multimodal model (AudioLM) that handles audio and text inputs, allowing you to have voice interactions without ASR modules. Qwen2-Audio supports English, Chinese, and major European languages,and provides voice chat and audio analysis capabilities for local use cases like:

Speaker identification and response
Speech translation and transcription
Mixed audio and noise detection
Music and sound analysis
Demo
See more demos in our blogs

How to Run Locally On Device
In the following, we demonstrate how to run Qwen2-Audio locally on your device.

Step 1: Install Nexa-SDK (local on-device inference framework)

Install Nexa-SDK

Nexa-SDK is a open-sourced, local on-device inference framework, supporting text generation, image generation, vision-language models (VLM), audio-language models, speech-to-text (ASR), and text-to-speech (TTS) capabilities. Installable via Python Package or Executable Installer.

Step 2: Then run the following code in your terminal

bash
nexa run qwen2audio
This will run default q4_K_M quantization.

For terminal:

Drag and drop your audio file into the terminal (or enter file path on Linux)
Add text prompt to guide analysis or leave empty for direct voice input
or to use with local UI (streamlit):

bash
nexa run qwen2audio -st
Choose Quantizations for your device
Run different quantization versions here and check RAM requirements in our list.

The default q4_K_M version requires 4.2GB of RAM.

Use Cases
Voice Chat
Answer daily questions
Offer suggestions
Speaker identification and response
Speech translation
Detecting background noise and responding accordingly
Audio Analysis
Information Extraction
Audio summary
Speech Transcription and Expansion
Mixed audio and noise detection
Music and sound analysis
Performance Benchmark
Results demonstrate that Qwen2-Audio significantly outperforms either previous SOTAs or Qwen-Audio across all tasks.

Blog
Learn more in our blogs

Join Community
Discord | X(Twitter)