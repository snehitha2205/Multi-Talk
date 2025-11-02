# # # Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# # import logging
# # import os
# # import sys
# # import json
# # import warnings
# # from datetime import datetime
# # from flask import Flask, request, jsonify, send_file
# # import uuid
# # import glob

# # warnings.filterwarnings('ignore')

# # import random

# # import torch
# # import torch.distributed as dist
# # from PIL import Image
# # import subprocess

# # import wan
# # from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
# # from wan.utils.utils import cache_image, cache_video, str2bool
# # from wan.utils.multitalk_utils import save_video_ffmpeg
# # from kokoro import KPipeline
# # from transformers import Wav2Vec2FeatureExtractor
# # from src.audio_analysis.wav2vec2 import Wav2Vec2Model

# # import librosa
# # import pyloudnorm as pyln
# # import numpy as np
# # from einops import rearrange
# # import soundfile as sf
# # import re

# # # Import ngrok
# # from pyngrok import ngrok

# # app = Flask(__name__)

# # # Configuration - USING YOUR PATHS FROM CLI
# # CONFIG = {
# #     "ckpt_dir": "/content/drive/MyDrive/weights/Wan2.1-I2V-14B-480P",
# #     "quant_dir": "/content/drive/MyDrive/weights/MeiGen-MultiTalk", 
# #     "wav2vec_dir": "/content/drive/MyDrive/weights/chinese-wav2vec2-base",
# #     "task": "multitalk-14B",
# #     "size": "multitalk-480",
# #     "frame_num": 81,
# #     "lora_dir": None,
# #     "lora_scale": [1.2],
# #     "offload_model": True,
# #     "ulysses_size": 1,
# #     "ring_size": 1,
# #     "t5_fsdp": False,
# #     "t5_cpu": False,
# #     "dit_fsdp": False,
# #     "audio_save_dir": "save_audio",
# #     "base_seed": 42,
# #     "motion_frame": 25,
# #     "mode": "streaming",
# #     "sample_steps": 40,
# #     "sample_shift": None,
# #     "sample_text_guide_scale": 5.0,
# #     "sample_audio_guide_scale": 4.0,
# #     "num_persistent_param_in_dit": None,
# #     "use_teacache": False,
# #     "teacache_thresh": 0.2,
# #     "use_apg": False,
# #     "apg_momentum": -0.75,
# #     "apg_norm_threshold": 55,
# #     "color_correction_strength": 1.0,
# #     "quant": "int8",
# #     "device": "cuda" if torch.cuda.is_available() else "cpu"
# # }

# # # Global variables for pre-loaded models
# # wan_pipeline = None
# # wav2vec_feature_extractor = None
# # audio_encoder = None
# # pipeline_initialized = False

# # def _validate_config(config):
# #     """Validate configuration matching CLI _validate_args functionality"""
# #     if config["ckpt_dir"] is None:
# #         raise ValueError("Please specify the checkpoint directory.")
# #     if config["task"] not in WAN_CONFIGS:
# #         raise ValueError(f"Unsupport task: {config['task']}")

# #     if config["sample_steps"] is None:
# #         config["sample_steps"] = 40

# #     if config["sample_shift"] is None:
# #         if config["size"] == 'multitalk-480':
# #             config["sample_shift"] = 7
# #         elif config["size"] == 'multitalk-720':
# #             config["sample_shift"] = 11
# #         else:
# #             config["sample_shift"] = 3.0

# #     config["base_seed"] = config["base_seed"] if config["base_seed"] >= 0 else random.randint(0, 99999999)
    
# #     if config["size"] not in SUPPORTED_SIZES[config["task"]]:
# #         raise ValueError(f"Unsupport size {config['size']} for task {config['task']}, supported sizes are: {', '.join(SUPPORTED_SIZES[config['task']])}")

# # def custom_init(device, wav2vec):    
# #     audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec, local_files_only=True).to(device)
# #     audio_encoder.feature_extractor._freeze_parameters()
# #     wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec, local_files_only=True)
# #     return wav2vec_feature_extractor, audio_encoder

# # def loudness_norm(audio_array, sr=16000, lufs=-23):
# #     meter = pyln.Meter(sr)
# #     loudness = meter.integrated_loudness(audio_array)
# #     if abs(loudness) > 100:
# #         return audio_array
# #     normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
# #     return normalized_audio

# # def get_embedding(speech_array, wav2vec_feature_extractor, audio_encoder, sr=16000, device='cpu'):
# #     audio_duration = len(speech_array) / sr
# #     video_length = audio_duration * 25

# #     audio_feature = np.squeeze(
# #         wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
# #     )
# #     audio_feature = torch.from_numpy(audio_feature).float().to(device=device)
# #     audio_feature = audio_feature.unsqueeze(0)

# #     with torch.no_grad():
# #         embeddings = audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)

# #     if len(embeddings) == 0:
# #         print("Fail to extract audio embedding")
# #         return None

# #     audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
# #     audio_emb = rearrange(audio_emb, "b s d -> s b d")
# #     audio_emb = audio_emb.cpu().detach()
# #     return audio_emb

# # def extract_audio_from_video(filename, sample_rate):
# #     raw_audio_path = filename.split('/')[-1].split('.')[0]+'.wav'
# #     ffmpeg_command = [
# #         "ffmpeg",
# #         "-y",
# #         "-i",
# #         str(filename),
# #         "-vn",
# #         "-acodec",
# #         "pcm_s16le",
# #         "-ar",
# #         "16000",
# #         "-ac",
# #         "2",
# #         str(raw_audio_path),
# #     ]
# #     subprocess.run(ffmpeg_command, check=True)
# #     human_speech_array, sr = librosa.load(raw_audio_path, sr=sample_rate)
# #     human_speech_array = loudness_norm(human_speech_array, sr)
# #     os.remove(raw_audio_path)
# #     return human_speech_array

# # def audio_prepare_single(audio_path, sample_rate=16000):
# #     ext = os.path.splitext(audio_path)[1].lower()
# #     if ext in ['.mp4', '.mov', '.avi', '.mkv']:
# #         human_speech_array = extract_audio_from_video(audio_path, sample_rate)
# #         return human_speech_array
# #     else:
# #         human_speech_array, sr = librosa.load(audio_path, sr=sample_rate)
# #         human_speech_array = loudness_norm(human_speech_array, sr)
# #         return human_speech_array

# # def process_tts_single(text, save_dir, voice1):    
# #     s1_sentences = []
# #     pipeline = KPipeline(lang_code='a', repo_id='/content/drive/MyDrive/weights/Kokoro-82M')

# #     voice_tensor = torch.load(voice1, weights_only=True)
# #     generator = pipeline(
# #         text, voice=voice_tensor,
# #         speed=1, split_pattern=r'\n+'
# #     )
# #     audios = []
# #     for i, (gs, ps, audio) in enumerate(generator):
# #         audios.append(audio)
# #     audios = torch.concat(audios, dim=0)
# #     s1_sentences.append(audios)
# #     s1_sentences = torch.concat(s1_sentences, dim=0)
# #     save_path1 = f'{save_dir}/s1.wav'
# #     sf.write(save_path1, s1_sentences, 24000)
# #     s1, _ = librosa.load(save_path1, sr=16000)
# #     return s1, save_path1

# # def process_tts_multi(text, save_dir, voice1, voice2):
# #     pattern = r'\(s(\d+)\)\s*(.?)(?=\s\(s\d+\)|$)'
# #     matches = re.findall(pattern, text, re.DOTALL)
    
# #     s1_sentences = []
# #     s2_sentences = []

# #     pipeline = KPipeline(lang_code='a', repo_id='/content/drive/MyDrive/weights/Kokoro-82M')
# #     for idx, (speaker, content) in enumerate(matches):
# #         if speaker == '1':
# #             voice_tensor = torch.load(voice1, weights_only=True)
# #             generator = pipeline(
# #                 content, voice=voice_tensor,
# #                 speed=1, split_pattern=r'\n+'
# #             )
# #             audios = []
# #             for i, (gs, ps, audio) in enumerate(generator):
# #                 audios.append(audio)
# #             audios = torch.concat(audios, dim=0)
# #             s1_sentences.append(audios)
# #             s2_sentences.append(torch.zeros_like(audios))
# #         elif speaker == '2':
# #             voice_tensor = torch.load(voice2, weights_only=True)
# #             generator = pipeline(
# #                 content, voice=voice_tensor,
# #                 speed=1, split_pattern=r'\n+'
# #             )
# #             audios = []
# #             for i, (gs, ps, audio) in enumerate(generator):
# #                 audios.append(audio)
# #             audios = torch.concat(audios, dim=0)
# #             s2_sentences.append(audios)
# #             s1_sentences.append(torch.zeros_like(audios))
    
# #     s1_sentences = torch.concat(s1_sentences, dim=0)
# #     s2_sentences = torch.concat(s2_sentences, dim=0)
# #     sum_sentences = s1_sentences + s2_sentences
# #     save_path1 = f'{save_dir}/s1.wav'
# #     save_path2 = f'{save_dir}/s2.wav'
# #     save_path_sum = f'{save_dir}/sum.wav'
# #     sf.write(save_path1, s1_sentences, 24000)
# #     sf.write(save_path2, s2_sentences, 24000)
# #     sf.write(save_path_sum, sum_sentences, 24000)

# #     s1, _ = librosa.load(save_path1, sr=16000)
# #     s2, _ = librosa.load(save_path2, sr=16000)
# #     return s1, s2, save_path_sum

# # def process_tts_triple(text, save_dir, voice1, voice2, voice3):
# #     pattern = r'\(s(\d+)\)\s*(.*?)(?=\s*\(s\d+\)|$)'
# #     matches = re.findall(pattern, text, re.DOTALL)
    
# #     s1_sentences = []
# #     s2_sentences = []
# #     s3_sentences = []

# #     pipeline = KPipeline(lang_code='a', repo_id='/content/drive/MyDrive/weights/Kokoro-82M')
    
# #     for idx, (speaker, content) in enumerate(matches):
# #         if speaker == '1':
# #             voice_tensor = torch.load(voice1, weights_only=True)
# #             generator = pipeline(
# #                 content, voice=voice_tensor,
# #                 speed=1, split_pattern=r'\n+'
# #             )
# #             audios = []
# #             for i, (gs, ps, audio) in enumerate(generator):
# #                 audios.append(audio)
# #             audios = torch.concat(audios, dim=0)
# #             s1_sentences.append(audios)
# #             s2_sentences.append(torch.zeros_like(audios))
# #             s3_sentences.append(torch.zeros_like(audios))
# #         elif speaker == '2':
# #             voice_tensor = torch.load(voice2, weights_only=True)
# #             generator = pipeline(
# #                 content, voice=voice_tensor,
# #                 speed=1, split_pattern=r'\n+'
# #             )
# #             audios = []
# #             for i, (gs, ps, audio) in enumerate(generator):
# #                 audios.append(audio)
# #             audios = torch.concat(audios, dim=0)
# #             s2_sentences.append(audios)
# #             s1_sentences.append(torch.zeros_like(audios))
# #             s3_sentences.append(torch.zeros_like(audios))
# #         elif speaker == '3':
# #             voice_tensor = torch.load(voice3, weights_only=True)
# #             generator = pipeline(
# #                 content, voice=voice_tensor,
# #                 speed=1, split_pattern=r'\n+'
# #             )
# #             audios = []
# #             for i, (gs, ps, audio) in enumerate(generator):
# #                 audios.append(audio)
# #             audios = torch.concat(audios, dim=0)
# #             s3_sentences.append(audios)
# #             s1_sentences.append(torch.zeros_like(audios))
# #             s2_sentences.append(torch.zeros_like(audios))
    
# #     s1_sentences = torch.concat(s1_sentences, dim=0)
# #     s2_sentences = torch.concat(s2_sentences, dim=0)
# #     s3_sentences = torch.concat(s3_sentences, dim=0)
# #     sum_sentences = s1_sentences + s2_sentences + s3_sentences
    
# #     save_path1 = f'{save_dir}/s1.wav'
# #     save_path2 = f'{save_dir}/s2.wav'
# #     save_path3 = f'{save_dir}/s3.wav'
# #     save_path_sum = f'{save_dir}/sum.wav'
    
# #     sf.write(save_path1, s1_sentences, 24000)
# #     sf.write(save_path2, s2_sentences, 24000)
# #     sf.write(save_path3, s3_sentences, 24000)
# #     sf.write(save_path_sum, sum_sentences, 24000)

# #     s1, _ = librosa.load(save_path1, sr=16000)
# #     s2, _ = librosa.load(save_path2, sr=16000)
# #     s3, _ = librosa.load(save_path3, sr=16000)
    
# #     return s1, s2, s3, save_path_sum

# # def initialize_models(config):
# #     """Initialize models once at server startup"""
# #     global wan_pipeline, wav2vec_feature_extractor, audio_encoder, pipeline_initialized
    
# #     if pipeline_initialized:
# #         return True
        
# #     try:
# #         logging.info("Initializing models...")
        
# #         # Set device - use GPU if available
# #         device = torch.device(config["device"])
# #         if device.type == 'cuda':
# #             torch.cuda.set_device(0)
        
# #         # Set environment variables for single GPU operation
# #         os.environ["RANK"] = "0"
# #         os.environ["WORLD_SIZE"] = "1"
# #         os.environ["LOCAL_RANK"] = "0"
# #         os.environ["MASTER_ADDR"] = "localhost"
# #         os.environ["MASTER_PORT"] = "12355"
        
# #         # Initialize distributed process group for single GPU
# #         if not dist.is_initialized():
# #             dist.init_process_group(
# #                 backend="nccl",
# #                 init_method="env://",
# #                 rank=0,
# #                 world_size=1
# #             )
        
# #         # Initialize Wav2Vec models on correct device
# #         logging.info("Loading Wav2Vec models...")
# #         wav2vec_feature_extractor, audio_encoder = custom_init(device, config["wav2vec_dir"])
        
# #         # Load Wan pipeline configuration
# #         cfg = WAN_CONFIGS[config["task"]]
        
# #         # Set offload_model if None (same logic as original)
# #         if config["offload_model"] is None:
# #             config["offload_model"] = False  # Single GPU, no offloading needed
# #             logging.info(f"offload_model is not specified, set to {config['offload_model']}.")
        
# #         logging.info(f"Creating MultiTalk pipeline for task: {config['task']}")
# #         wan_pipeline = wan.MultiTalkPipeline(
# #             config=cfg,
# #             checkpoint_dir=config["ckpt_dir"],
# #             quant_dir=config["quant_dir"],
# #             device_id=0 if device.type == 'cuda' else -1,
# #             rank=0,
# #             t5_fsdp=config["t5_fsdp"],
# #             dit_fsdp=config["dit_fsdp"], 
# #             use_usp=False,  
# #             t5_cpu=config["t5_cpu"],
# #             lora_dir=config["lora_dir"],
# #             lora_scales=config["lora_scale"],
# #             quant=config["quant"]
# #         )

# #         if config["num_persistent_param_in_dit"] is not None:
# #             wan_pipeline.vram_management = True
# #             wan_pipeline.enable_vram_management(
# #                 num_persistent_param_in_dit=config["num_persistent_param_in_dit"]
# #             )
        
# #         pipeline_initialized = True
# #         logging.info("All models initialized successfully!")
# #         return True
        
# #     except Exception as e:
# #         logging.error(f"Failed to initialize models: {str(e)}")
# #         return False

# # def generate_video_with_audio_files(input_data, config, job_id):
# #     """Generate video using local audio files"""
# #     global wan_pipeline, wav2vec_feature_extractor, audio_encoder
    
# #     if not pipeline_initialized:
# #         raise RuntimeError("Models not initialized. Please restart the server.")
    
# #     try:
# #         # Process audio data from local files
# #         audio_save_dir = os.path.join(config["audio_save_dir"], job_id)
# #         os.makedirs(audio_save_dir, exist_ok=True)
        
# #         # Audio processing for local files
# #         num_speakers = len(input_data['cond_audio'])
# #         all_audio_arrays = []
# #         all_embeddings = []
# #         all_audio_paths = []

# #         device = torch.device(config["device"])
        
# #         for i in range(num_speakers):
# #             key = f'person{i+1}'
# #             audio_path = input_data['cond_audio'].get(key)
# #             if audio_path is not None:
# #                 speech = audio_prepare_single(audio_path)
# #                 all_audio_arrays.append(speech)
# #                 emb = get_embedding(speech, wav2vec_feature_extractor, audio_encoder, device=device)
# #                 emb_path = os.path.join(audio_save_dir, f'{i+1}.pt')
# #                 torch.save(emb, emb_path)
# #                 input_data['cond_audio'][key] = emb_path
# #                 all_embeddings.append(emb)
# #                 all_audio_paths.append(audio_path)

# #         if len(all_audio_arrays) > 0:
# #             max_len = max([len(a) for a in all_audio_arrays])
# #             padded = [np.pad(a, (0, max_len - len(a))) for a in all_audio_arrays]
# #             sum_audio = np.sum(padded, axis=0)
# #             sum_audio_path = os.path.join(audio_save_dir, 'sum.wav')
# #             sf.write(sum_audio_path, sum_audio, 16000)
# #             input_data['video_audio'] = sum_audio_path

# #         # Generate video using pre-loaded pipeline
# #         logging.info("Generating video with local audio files...")
# #         video = wan_pipeline.generate(
# #             input_data,
# #             size_buckget=config["size"],
# #             motion_frame=config["motion_frame"],
# #             frame_num=config["frame_num"],
# #             shift=config["sample_shift"],
# #             sampling_steps=config["sample_steps"],
# #             text_guide_scale=config["sample_text_guide_scale"],
# #             audio_guide_scale=config["sample_audio_guide_scale"],
# #             seed=config["base_seed"],
# #             offload_model=config["offload_model"],
# #             max_frames_num=config["frame_num"] if config["mode"] == 'clip' else 1000,
# #             color_correction_strength=config["color_correction_strength"],
# #             extra_args=config,
# #         )

# #         # Save video file
# #         formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
# #         formatted_prompt = input_data['prompt'].replace(" ", "_").replace("/", "_")[:50]
# #         save_file = f"{config['task']}_{config['size']}_{formatted_prompt}_{formatted_time}"
        
# #         output_path = f"{save_file}.mp4"
# #         logging.info(f"Saving generated video to {output_path}")
# #         save_video_ffmpeg(video, save_file, [input_data['video_audio']], high_quality_save=False)
        
# #         return output_path
        
# #     except Exception as e:
# #         logging.error(f"Video generation with local audio failed: {str(e)}")
# #         raise

# # def generate_video_with_tts(input_data, config, job_id):
# #     """Generate video using TTS"""
# #     global wan_pipeline, wav2vec_feature_extractor, audio_encoder
    
# #     if not pipeline_initialized:
# #         raise RuntimeError("Models not initialized. Please restart the server.")
    
# #     try:
# #         # Process TTS data
# #         audio_save_dir = os.path.join(config["audio_save_dir"], job_id)
# #         os.makedirs(audio_save_dir, exist_ok=True)
        
# #         device = torch.device(config["device"])
        
# #         # TTS processing
# #         if 'human2_voice' not in input_data['tts_audio'].keys():
# #             # Single speaker TTS
# #             new_human_speech1, sum_audio = process_tts_single(
# #                 input_data['tts_audio']['text'], 
# #                 audio_save_dir, 
# #                 input_data['tts_audio']['human1_voice']
# #             )
# #             audio_embedding_1 = get_embedding(new_human_speech1, wav2vec_feature_extractor, audio_encoder, device=device)
# #             emb1_path = os.path.join(audio_save_dir, '1.pt')
# #             torch.save(audio_embedding_1, emb1_path)
# #             input_data['cond_audio']['person1'] = emb1_path
# #             input_data['video_audio'] = sum_audio
            
# #         elif 'human3_voice' in input_data['tts_audio'].keys() and input_data['tts_audio']['human3_voice']:
# #             # Three speaker TTS
# #             new_human_speech1, new_human_speech2, new_human_speech3, sum_audio = process_tts_triple(
# #                 input_data['tts_audio']['text'], 
# #                 audio_save_dir, 
# #                 input_data['tts_audio']['human1_voice'], 
# #                 input_data['tts_audio']['human2_voice'],
# #                 input_data['tts_audio']['human3_voice']
# #             )
# #             audio_embedding_1 = get_embedding(new_human_speech1, wav2vec_feature_extractor, audio_encoder, device=device)
# #             audio_embedding_2 = get_embedding(new_human_speech2, wav2vec_feature_extractor, audio_encoder, device=device)
# #             audio_embedding_3 = get_embedding(new_human_speech3, wav2vec_feature_extractor, audio_encoder, device=device)
# #             emb1_path = os.path.join(audio_save_dir, '1.pt')
# #             emb2_path = os.path.join(audio_save_dir, '2.pt')
# #             emb3_path = os.path.join(audio_save_dir, '3.pt')
# #             torch.save(audio_embedding_1, emb1_path)
# #             torch.save(audio_embedding_2, emb2_path)
# #             torch.save(audio_embedding_3, emb3_path)
# #             input_data['cond_audio']['person1'] = emb1_path
# #             input_data['cond_audio']['person2'] = emb2_path
# #             input_data['cond_audio']['person3'] = emb3_path
# #             input_data['video_audio'] = sum_audio
# #         else:
# #             # Two speaker TTS
# #             new_human_speech1, new_human_speech2, sum_audio = process_tts_multi(
# #                 input_data['tts_audio']['text'], 
# #                 audio_save_dir, 
# #                 input_data['tts_audio']['human1_voice'], 
# #                 input_data['tts_audio']['human2_voice']
# #             )
# #             audio_embedding_1 = get_embedding(new_human_speech1, wav2vec_feature_extractor, audio_encoder, device=device)
# #             audio_embedding_2 = get_embedding(new_human_speech2, wav2vec_feature_extractor, audio_encoder, device=device)
# #             emb1_path = os.path.join(audio_save_dir, '1.pt')
# #             emb2_path = os.path.join(audio_save_dir, '2.pt')
# #             torch.save(audio_embedding_1, emb1_path)
# #             torch.save(audio_embedding_2, emb2_path)
# #             input_data['cond_audio']['person1'] = emb1_path
# #             input_data['cond_audio']['person2'] = emb2_path
# #             input_data['video_audio'] = sum_audio

# #         # Generate video using pre-loaded pipeline
# #         logging.info("Generating video with TTS...")
# #         video = wan_pipeline.generate(
# #             input_data,
# #             size_buckget=config["size"],
# #             motion_frame=config["motion_frame"],
# #             frame_num=config["frame_num"],
# #             shift=config["sample_shift"],
# #             sampling_steps=config["sample_steps"],
# #             text_guide_scale=config["sample_text_guide_scale"],
# #             audio_guide_scale=config["sample_audio_guide_scale"],
# #             seed=config["base_seed"],
# #             offload_model=config["offload_model"],
# #             max_frames_num=config["frame_num"] if config["mode"] == 'clip' else 1000,
# #             color_correction_strength=config["color_correction_strength"],
# #             extra_args=config,
# #         )

# #         # Save video file
# #         formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
# #         formatted_prompt = input_data['prompt'].replace(" ", "_").replace("/", "_")[:50]
# #         save_file = f"{config['task']}_{config['size']}_{formatted_prompt}_{formatted_time}"
        
# #         output_path = f"{save_file}.mp4"
# #         logging.info(f"Saving generated video to {output_path}")
# #         save_video_ffmpeg(video, save_file, [input_data['video_audio']], high_quality_save=False)
        
# #         return output_path
        
# #     except Exception as e:
# #         logging.error(f"Video generation with TTS failed: {str(e)}")
# #         raise

# # @app.route('/generate/with-audio', methods=['POST'])
# # def generate_with_audio_endpoint():
# #     """Endpoint for generating video with local audio files"""
# #     try:
# #         input_data = request.get_json()
        
# #         if not input_data:
# #             return jsonify({"error": "No JSON data provided"}), 400
        
# #         # Validate required fields for audio files
# #         required_fields = ['prompt', 'cond_image', 'cond_audio']
# #         for field in required_fields:
# #             if field not in input_data:
# #                 return jsonify({"error": f"Missing required field: {field}"}), 400
        
# #         job_id = str(uuid.uuid4())
# #         logging.info(f"Starting video generation with local audio - Job: {job_id}")
        
# #         # Generate video with local audio files
# #         output_file = generate_video_with_audio_files(input_data, CONFIG, job_id)
        
# #         if output_file and os.path.exists(output_file):
# #             return jsonify({
# #                 "status": "success",
# #                 "job_id": job_id,
# #                 "output_file": output_file,
# #                 "download_url": f"/download/{os.path.basename(output_file)}"
# #             }), 200
# #         else:
# #             return jsonify({
# #                 "status": "error",
# #                 "message": "Video generation failed - output file not created"
# #             }), 500
            
# #     except Exception as e:
# #         logging.error(f"Error in video generation with audio: {str(e)}")
# #         return jsonify({
# #             "status": "error",
# #             "message": str(e)
# #         }), 500

# # @app.route('/generate/with-tts', methods=['POST'])
# # def generate_with_tts_endpoint():
# #     """Endpoint for generating video with TTS"""
# #     try:
# #         input_data = request.get_json()
        
# #         if not input_data:
# #             return jsonify({"error": "No JSON data provided"}), 400
        
# #         # Validate required fields for TTS
# #         required_fields = ['prompt', 'cond_image', 'tts_audio']
# #         for field in required_fields:
# #             if field not in input_data:
# #                 return jsonify({"error": f"Missing required field: {field}"}), 400
        
# #         job_id = str(uuid.uuid4())
# #         logging.info(f"Starting video generation with TTS - Job: {job_id}")
        
# #         # Generate video with TTS
# #         output_file = generate_video_with_tts(input_data, CONFIG, job_id)
        
# #         if output_file and os.path.exists(output_file):
# #             return jsonify({
# #                 "status": "success",
# #                 "job_id": job_id,
# #                 "output_file": output_file,
# #                 "download_url": f"/download/{os.path.basename(output_file)}"
# #             }), 200
# #         else:
# #             return jsonify({
# #                 "status": "error",
# #                 "message": "Video generation failed - output file not created"
# #             }), 500
            
# #     except Exception as e:
# #         logging.error(f"Error in video generation with TTS: {str(e)}")
# #         return jsonify({
# #             "status": "error",
# #             "message": str(e)
# #         }), 500

# # @app.route('/download/<filename>', methods=['GET'])
# # def download_video(filename):
# #     try:
# #         if os.path.exists(filename):
# #             return send_file(filename, as_attachment=True)
# #         else:
# #             matches = glob.glob(f"*{filename}*")
# #             if matches:
# #                 return send_file(matches[0], as_attachment=True)
# #             else:
# #                 return jsonify({"error": "File not found"}), 404
# #     except Exception as e:
# #         return jsonify({"error": str(e)}), 500

# # @app.route('/health', methods=['GET'])
# # def health_check():
# #     status = "healthy" if pipeline_initialized else "models_not_loaded"
# #     return jsonify({
# #         "status": status,
# #         "models_initialized": pipeline_initialized
# #     }), 200

# # @app.route('/config', methods=['GET'])
# # def get_config():
# #     """Endpoint to get current configuration"""
# #     return jsonify({
# #         "config": CONFIG,
# #         "models_initialized": pipeline_initialized
# #     }), 200

# # def cleanup():
# #     """Cleanup function to properly shutdown distributed process group"""
# #     if dist.is_initialized():
# #         dist.destroy_process_group()

# # if __name__ == "__main__":
# #     # Set up logging
# #     logging.basicConfig(
# #         level=logging.INFO,
# #         format="[%(asctime)s] %(levelname)s: %(message)s",
# #         handlers=[logging.StreamHandler(stream=sys.stdout)]
# #     )
    
# #     # Validate configuration
# #     try:
# #         _validate_config(CONFIG)
# #         logging.info("Configuration validated successfully")
# #     except Exception as e:
# #         logging.error(f"Configuration error: {e}")
# #         sys.exit(1)
    
# #     # Initialize models at startup
# #     if initialize_models(CONFIG):
# #         logging.info("Flask server starting with pre-loaded models...")
        
# #         # Start ngrok tunnel
# #         public_url = ngrok.connect(5000)
# #         logging.info(f"🚀 Public URL created: {public_url}")
# #         logging.info("📋 Use this URL in Postman to test the API:")
# #         logging.info(f"   Health Check: GET {public_url}/health")
# #         logging.info(f"   TTS Generation: POST {public_url}/generate/with-tts")
# #         logging.info(f"   Audio Generation: POST {public_url}/generate/with-audio")
        
# #         try:
# #             app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)
# #         finally:
# #             cleanup()
# #             ngrok.kill()  # Close ngrok tunnel when server stops
# #     else:
# #         logging.error("Failed to initialize models. Server cannot start.")
# #         sys.exit(1)



# # Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# import logging
# import os
# import sys
# import json
# import warnings
# from datetime import datetime
# from flask import Flask, request, jsonify, send_file
# import uuid
# import glob

# warnings.filterwarnings('ignore')

# import random

# import torch
# import torch.distributed as dist
# from PIL import Image
# import subprocess

# import wan
# from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
# from wan.utils.utils import cache_image, cache_video, str2bool
# from wan.utils.multitalk_utils import save_video_ffmpeg
# from kokoro import KPipeline
# from transformers import Wav2Vec2FeatureExtractor
# from src.audio_analysis.wav2vec2 import Wav2Vec2Model

# import librosa
# import pyloudnorm as pyln
# import numpy as np
# from einops import rearrange
# import soundfile as sf
# import re

# app = Flask(__name__)

# # Configuration - USING YOUR PATHS FROM CLI
# CONFIG = {
#     "ckpt_dir": "/content/drive/MyDrive/weights/Wan2.1-I2V-14B-480P",
#     "quant_dir": "/content/drive/MyDrive/weights/MeiGen-MultiTalk", 
#     "wav2vec_dir": "/content/drive/MyDrive/weights/chinese-wav2vec2-base",
#     "task": "multitalk-14B",
#     "size": "multitalk-480",
#     "frame_num": 81,
#     "lora_dir": None,
#     "lora_scale": [1.2],
#     "offload_model": True,
#     "ulysses_size": 1,
#     "ring_size": 1,
#     "t5_fsdp": False,
#     "t5_cpu": False,
#     "dit_fsdp": False,
#     "audio_save_dir": "save_audio",
#     "base_seed": 42,
#     "motion_frame": 25,
#     "mode": "streaming",
#     "sample_steps": 40,
#     "sample_shift": None,
#     "sample_text_guide_scale": 5.0,
#     "sample_audio_guide_scale": 4.0,
#     "num_persistent_param_in_dit": None,
#     "use_teacache": False,
#     "teacache_thresh": 0.2,
#     "use_apg": False,
#     "apg_momentum": -0.75,
#     "apg_norm_threshold": 55,
#     "color_correction_strength": 1.0,
#     "quant": "int8",
#     "device": "cuda" if torch.cuda.is_available() else "cpu"
# }

# # Global variables for pre-loaded models
# wan_pipeline = None
# wav2vec_feature_extractor = None
# audio_encoder = None
# pipeline_initialized = False

# def _validate_config(config):
#     """Validate configuration matching CLI _validate_args functionality"""
#     if config["ckpt_dir"] is None:
#         raise ValueError("Please specify the checkpoint directory.")
#     if config["task"] not in WAN_CONFIGS:
#         raise ValueError(f"Unsupport task: {config['task']}")

#     if config["sample_steps"] is None:
#         config["sample_steps"] = 40

#     if config["sample_shift"] is None:
#         if config["size"] == 'multitalk-480':
#             config["sample_shift"] = 7
#         elif config["size"] == 'multitalk-720':
#             config["sample_shift"] = 11
#         else:
#             config["sample_shift"] = 3.0

#     config["base_seed"] = config["base_seed"] if config["base_seed"] >= 0 else random.randint(0, 99999999)
    
#     if config["size"] not in SUPPORTED_SIZES[config["task"]]:
#         raise ValueError(f"Unsupport size {config['size']} for task {config['task']}, supported sizes are: {', '.join(SUPPORTED_SIZES[config['task']])}")

# def custom_init(device, wav2vec):    
#     audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec, local_files_only=True).to(device)
#     audio_encoder.feature_extractor._freeze_parameters()
#     wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec, local_files_only=True)
#     return wav2vec_feature_extractor, audio_encoder

# def loudness_norm(audio_array, sr=16000, lufs=-23):
#     meter = pyln.Meter(sr)
#     loudness = meter.integrated_loudness(audio_array)
#     if abs(loudness) > 100:
#         return audio_array
#     normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
#     return normalized_audio

# def get_embedding(speech_array, wav2vec_feature_extractor, audio_encoder, sr=16000, device='cpu'):
#     audio_duration = len(speech_array) / sr
#     video_length = audio_duration * 25

#     audio_feature = np.squeeze(
#         wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
#     )
#     audio_feature = torch.from_numpy(audio_feature).float().to(device=device)
#     audio_feature = audio_feature.unsqueeze(0)

#     with torch.no_grad():
#         embeddings = audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)

#     if len(embeddings) == 0:
#         print("Fail to extract audio embedding")
#         return None

#     audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
#     audio_emb = rearrange(audio_emb, "b s d -> s b d")
#     audio_emb = audio_emb.cpu().detach()
#     return audio_emb

# def extract_audio_from_video(filename, sample_rate):
#     raw_audio_path = filename.split('/')[-1].split('.')[0]+'.wav'
#     ffmpeg_command = [
#         "ffmpeg",
#         "-y",
#         "-i",
#         str(filename),
#         "-vn",
#         "-acodec",
#         "pcm_s16le",
#         "-ar",
#         "16000",
#         "-ac",
#         "2",
#         str(raw_audio_path),
#     ]
#     subprocess.run(ffmpeg_command, check=True)
#     human_speech_array, sr = librosa.load(raw_audio_path, sr=sample_rate)
#     human_speech_array = loudness_norm(human_speech_array, sr)
#     os.remove(raw_audio_path)
#     return human_speech_array

# def audio_prepare_single(audio_path, sample_rate=16000):
#     ext = os.path.splitext(audio_path)[1].lower()
#     if ext in ['.mp4', '.mov', '.avi', '.mkv']:
#         human_speech_array = extract_audio_from_video(audio_path, sample_rate)
#         return human_speech_array
#     else:
#         human_speech_array, sr = librosa.load(audio_path, sr=sample_rate)
#         human_speech_array = loudness_norm(human_speech_array, sr)
#         return human_speech_array

# def process_tts_single(text, save_dir, voice1):    
#     s1_sentences = []
#     pipeline = KPipeline(lang_code='a', repo_id='/content/drive/MyDrive/weights/Kokoro-82M')

#     voice_tensor = torch.load(voice1, weights_only=True)
#     generator = pipeline(
#         text, voice=voice_tensor,
#         speed=1, split_pattern=r'\n+'
#     )
#     audios = []
#     for i, (gs, ps, audio) in enumerate(generator):
#         audios.append(audio)
#     audios = torch.concat(audios, dim=0)
#     s1_sentences.append(audios)
#     s1_sentences = torch.concat(s1_sentences, dim=0)
#     save_path1 = f'{save_dir}/s1.wav'
#     sf.write(save_path1, s1_sentences, 24000)
#     s1, _ = librosa.load(save_path1, sr=16000)
#     return s1, save_path1

# def process_tts_multi(text, save_dir, voice1, voice2):
#     pattern = r'\(s(\d+)\)\s*(.?)(?=\s\(s\d+\)|$)'
#     matches = re.findall(pattern, text, re.DOTALL)
    
#     s1_sentences = []
#     s2_sentences = []

#     pipeline = KPipeline(lang_code='a', repo_id='/content/drive/MyDrive/weights/Kokoro-82M')
#     for idx, (speaker, content) in enumerate(matches):
#         if speaker == '1':
#             voice_tensor = torch.load(voice1, weights_only=True)
#             generator = pipeline(
#                 content, voice=voice_tensor,
#                 speed=1, split_pattern=r'\n+'
#             )
#             audios = []
#             for i, (gs, ps, audio) in enumerate(generator):
#                 audios.append(audio)
#             audios = torch.concat(audios, dim=0)
#             s1_sentences.append(audios)
#             s2_sentences.append(torch.zeros_like(audios))
#         elif speaker == '2':
#             voice_tensor = torch.load(voice2, weights_only=True)
#             generator = pipeline(
#                 content, voice=voice_tensor,
#                 speed=1, split_pattern=r'\n+'
#             )
#             audios = []
#             for i, (gs, ps, audio) in enumerate(generator):
#                 audios.append(audio)
#             audios = torch.concat(audios, dim=0)
#             s2_sentences.append(audios)
#             s1_sentences.append(torch.zeros_like(audios))
    
#     s1_sentences = torch.concat(s1_sentences, dim=0)
#     s2_sentences = torch.concat(s2_sentences, dim=0)
#     sum_sentences = s1_sentences + s2_sentences
#     save_path1 = f'{save_dir}/s1.wav'
#     save_path2 = f'{save_dir}/s2.wav'
#     save_path_sum = f'{save_dir}/sum.wav'
#     sf.write(save_path1, s1_sentences, 24000)
#     sf.write(save_path2, s2_sentences, 24000)
#     sf.write(save_path_sum, sum_sentences, 24000)

#     s1, _ = librosa.load(save_path1, sr=16000)
#     s2, _ = librosa.load(save_path2, sr=16000)
#     return s1, s2, save_path_sum

# def process_tts_triple(text, save_dir, voice1, voice2, voice3):
#     pattern = r'\(s(\d+)\)\s*(.*?)(?=\s*\(s\d+\)|$)'
#     matches = re.findall(pattern, text, re.DOTALL)
    
#     s1_sentences = []
#     s2_sentences = []
#     s3_sentences = []

#     pipeline = KPipeline(lang_code='a', repo_id='/content/drive/MyDrive/weights/Kokoro-82M')
    
#     for idx, (speaker, content) in enumerate(matches):
#         if speaker == '1':
#             voice_tensor = torch.load(voice1, weights_only=True)
#             generator = pipeline(
#                 content, voice=voice_tensor,
#                 speed=1, split_pattern=r'\n+'
#             )
#             audios = []
#             for i, (gs, ps, audio) in enumerate(generator):
#                 audios.append(audio)
#             audios = torch.concat(audios, dim=0)
#             s1_sentences.append(audios)
#             s2_sentences.append(torch.zeros_like(audios))
#             s3_sentences.append(torch.zeros_like(audios))
#         elif speaker == '2':
#             voice_tensor = torch.load(voice2, weights_only=True)
#             generator = pipeline(
#                 content, voice=voice_tensor,
#                 speed=1, split_pattern=r'\n+'
#             )
#             audios = []
#             for i, (gs, ps, audio) in enumerate(generator):
#                 audios.append(audio)
#             audios = torch.concat(audios, dim=0)
#             s2_sentences.append(audios)
#             s1_sentences.append(torch.zeros_like(audios))
#             s3_sentences.append(torch.zeros_like(audios))
#         elif speaker == '3':
#             voice_tensor = torch.load(voice3, weights_only=True)
#             generator = pipeline(
#                 content, voice=voice_tensor,
#                 speed=1, split_pattern=r'\n+'
#             )
#             audios = []
#             for i, (gs, ps, audio) in enumerate(generator):
#                 audios.append(audio)
#             audios = torch.concat(audios, dim=0)
#             s3_sentences.append(audios)
#             s1_sentences.append(torch.zeros_like(audios))
#             s2_sentences.append(torch.zeros_like(audios))
    
#     s1_sentences = torch.concat(s1_sentences, dim=0)
#     s2_sentences = torch.concat(s2_sentences, dim=0)
#     s3_sentences = torch.concat(s3_sentences, dim=0)
#     sum_sentences = s1_sentences + s2_sentences + s3_sentences
    
#     save_path1 = f'{save_dir}/s1.wav'
#     save_path2 = f'{save_dir}/s2.wav'
#     save_path3 = f'{save_dir}/s3.wav'
#     save_path_sum = f'{save_dir}/sum.wav'
    
#     sf.write(save_path1, s1_sentences, 24000)
#     sf.write(save_path2, s2_sentences, 24000)
#     sf.write(save_path3, s3_sentences, 24000)
#     sf.write(save_path_sum, sum_sentences, 24000)

#     s1, _ = librosa.load(save_path1, sr=16000)
#     s2, _ = librosa.load(save_path2, sr=16000)
#     s3, _ = librosa.load(save_path3, sr=16000)
    
#     return s1, s2, s3, save_path_sum

# def initialize_models(config):
#     """Initialize models once at server startup"""
#     global wan_pipeline, wav2vec_feature_extractor, audio_encoder, pipeline_initialized
    
#     if pipeline_initialized:
#         return True
        
#     try:
#         logging.info("Initializing models...")
        
#         # Set device - use GPU if available
#         device = torch.device(config["device"])
#         if device.type == 'cuda':
#             torch.cuda.set_device(0)
        
#         # Set environment variables for single GPU operation
#         os.environ["RANK"] = "0"
#         os.environ["WORLD_SIZE"] = "1"
#         os.environ["LOCAL_RANK"] = "0"
#         os.environ["MASTER_ADDR"] = "localhost"
#         os.environ["MASTER_PORT"] = "12355"
        
#         # Initialize distributed process group for single GPU
#         if not dist.is_initialized():
#             dist.init_process_group(
#                 backend="nccl",
#                 init_method="env://",
#                 rank=0,
#                 world_size=1
#             )
        
#         # Initialize Wav2Vec models on correct device
#         logging.info("Loading Wav2Vec models...")
#         wav2vec_feature_extractor, audio_encoder = custom_init(device, config["wav2vec_dir"])
        
#         # Load Wan pipeline configuration
#         cfg = WAN_CONFIGS[config["task"]]
        
#         # Set offload_model if None (same logic as original)
#         if config["offload_model"] is None:
#             config["offload_model"] = False  # Single GPU, no offloading needed
#             logging.info(f"offload_model is not specified, set to {config['offload_model']}.")
        
#         logging.info(f"Creating MultiTalk pipeline for task: {config['task']}")
#         wan_pipeline = wan.MultiTalkPipeline(
#             config=cfg,
#             checkpoint_dir=config["ckpt_dir"],
#             quant_dir=config["quant_dir"],
#             device_id=0 if device.type == 'cuda' else -1,
#             rank=0,
#             t5_fsdp=config["t5_fsdp"],
#             dit_fsdp=config["dit_fsdp"], 
#             use_usp=False,  
#             t5_cpu=config["t5_cpu"],
#             lora_dir=config["lora_dir"],
#             lora_scales=config["lora_scale"],
#             quant=config["quant"]
#         )

#         if config["num_persistent_param_in_dit"] is not None:
#             wan_pipeline.vram_management = True
#             wan_pipeline.enable_vram_management(
#                 num_persistent_param_in_dit=config["num_persistent_param_in_dit"]
#             )
        
#         pipeline_initialized = True
#         logging.info("All models initialized successfully!")
#         return True
        
#     except Exception as e:
#         logging.error(f"Failed to initialize models: {str(e)}")
#         return False

# def generate_video_with_audio_files(input_data, config, job_id):
#     """Generate video using local audio files"""
#     global wan_pipeline, wav2vec_feature_extractor, audio_encoder
    
#     if not pipeline_initialized:
#         raise RuntimeError("Models not initialized. Please restart the server.")
    
#     try:
#         # Process audio data from local files
#         audio_save_dir = os.path.join(config["audio_save_dir"], job_id)
#         os.makedirs(audio_save_dir, exist_ok=True)
        
#         # Audio processing for local files
#         num_speakers = len(input_data['cond_audio'])
#         all_audio_arrays = []
#         all_embeddings = []
#         all_audio_paths = []

#         device = torch.device(config["device"])
        
#         for i in range(num_speakers):
#             key = f'person{i+1}'
#             audio_path = input_data['cond_audio'].get(key)
#             if audio_path is not None:
#                 speech = audio_prepare_single(audio_path)
#                 all_audio_arrays.append(speech)
#                 emb = get_embedding(speech, wav2vec_feature_extractor, audio_encoder, device=device)
#                 emb_path = os.path.join(audio_save_dir, f'{i+1}.pt')
#                 torch.save(emb, emb_path)
#                 input_data['cond_audio'][key] = emb_path
#                 all_embeddings.append(emb)
#                 all_audio_paths.append(audio_path)

#         if len(all_audio_arrays) > 0:
#             max_len = max([len(a) for a in all_audio_arrays])
#             padded = [np.pad(a, (0, max_len - len(a))) for a in all_audio_arrays]
#             sum_audio = np.sum(padded, axis=0)
#             sum_audio_path = os.path.join(audio_save_dir, 'sum.wav')
#             sf.write(sum_audio_path, sum_audio, 16000)
#             input_data['video_audio'] = sum_audio_path

#         # Generate video using pre-loaded pipeline
#         logging.info("Generating video with local audio files...")
#         video = wan_pipeline.generate(
#             input_data,
#             size_buckget=config["size"],
#             motion_frame=config["motion_frame"],
#             frame_num=config["frame_num"],
#             shift=config["sample_shift"],
#             sampling_steps=config["sample_steps"],
#             text_guide_scale=config["sample_text_guide_scale"],
#             audio_guide_scale=config["sample_audio_guide_scale"],
#             seed=config["base_seed"],
#             offload_model=config["offload_model"],
#             max_frames_num=config["frame_num"] if config["mode"] == 'clip' else 1000,
#             color_correction_strength=config["color_correction_strength"],
#             extra_args=config,
#         )

#         # Save video file
#         formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
#         formatted_prompt = input_data['prompt'].replace(" ", "_").replace("/", "_")[:50]
#         save_file = f"{config['task']}_{config['size']}_{formatted_prompt}_{formatted_time}"
        
#         output_path = f"{save_file}.mp4"
#         logging.info(f"Saving generated video to {output_path}")
#         save_video_ffmpeg(video, save_file, [input_data['video_audio']], high_quality_save=False)
        
#         return output_path
        
#     except Exception as e:
#         logging.error(f"Video generation with local audio failed: {str(e)}")
#         raise

# def generate_video_with_tts(input_data, config, job_id):
#     """Generate video using TTS"""
#     global wan_pipeline, wav2vec_feature_extractor, audio_encoder
    
#     if not pipeline_initialized:
#         raise RuntimeError("Models not initialized. Please restart the server.")
    
#     try:
#         # Process TTS data
#         audio_save_dir = os.path.join(config["audio_save_dir"], job_id)
#         os.makedirs(audio_save_dir, exist_ok=True)
        
#         device = torch.device(config["device"])
        
#         # TTS processing
#         if 'human2_voice' not in input_data['tts_audio'].keys():
#             # Single speaker TTS
#             new_human_speech1, sum_audio = process_tts_single(
#                 input_data['tts_audio']['text'], 
#                 audio_save_dir, 
#                 input_data['tts_audio']['human1_voice']
#             )
#             audio_embedding_1 = get_embedding(new_human_speech1, wav2vec_feature_extractor, audio_encoder, device=device)
#             emb1_path = os.path.join(audio_save_dir, '1.pt')
#             torch.save(audio_embedding_1, emb1_path)
#             input_data['cond_audio']['person1'] = emb1_path
#             input_data['video_audio'] = sum_audio
            
#         elif 'human3_voice' in input_data['tts_audio'].keys() and input_data['tts_audio']['human3_voice']:
#             # Three speaker TTS
#             new_human_speech1, new_human_speech2, new_human_speech3, sum_audio = process_tts_triple(
#                 input_data['tts_audio']['text'], 
#                 audio_save_dir, 
#                 input_data['tts_audio']['human1_voice'], 
#                 input_data['tts_audio']['human2_voice'],
#                 input_data['tts_audio']['human3_voice']
#             )
#             audio_embedding_1 = get_embedding(new_human_speech1, wav2vec_feature_extractor, audio_encoder, device=device)
#             audio_embedding_2 = get_embedding(new_human_speech2, wav2vec_feature_extractor, audio_encoder, device=device)
#             audio_embedding_3 = get_embedding(new_human_speech3, wav2vec_feature_extractor, audio_encoder, device=device)
#             emb1_path = os.path.join(audio_save_dir, '1.pt')
#             emb2_path = os.path.join(audio_save_dir, '2.pt')
#             emb3_path = os.path.join(audio_save_dir, '3.pt')
#             torch.save(audio_embedding_1, emb1_path)
#             torch.save(audio_embedding_2, emb2_path)
#             torch.save(audio_embedding_3, emb3_path)
#             input_data['cond_audio']['person1'] = emb1_path
#             input_data['cond_audio']['person2'] = emb2_path
#             input_data['cond_audio']['person3'] = emb3_path
#             input_data['video_audio'] = sum_audio
#         else:
#             # Two speaker TTS
#             new_human_speech1, new_human_speech2, sum_audio = process_tts_multi(
#                 input_data['tts_audio']['text'], 
#                 audio_save_dir, 
#                 input_data['tts_audio']['human1_voice'], 
#                 input_data['tts_audio']['human2_voice']
#             )
#             audio_embedding_1 = get_embedding(new_human_speech1, wav2vec_feature_extractor, audio_encoder, device=device)
#             audio_embedding_2 = get_embedding(new_human_speech2, wav2vec_feature_extractor, audio_encoder, device=device)
#             emb1_path = os.path.join(audio_save_dir, '1.pt')
#             emb2_path = os.path.join(audio_save_dir, '2.pt')
#             torch.save(audio_embedding_1, emb1_path)
#             torch.save(audio_embedding_2, emb2_path)
#             input_data['cond_audio']['person1'] = emb1_path
#             input_data['cond_audio']['person2'] = emb2_path
#             input_data['video_audio'] = sum_audio

#         # Generate video using pre-loaded pipeline
#         logging.info("Generating video with TTS...")
#         video = wan_pipeline.generate(
#             input_data,
#             size_buckget=config["size"],
#             motion_frame=config["motion_frame"],
#             frame_num=config["frame_num"],
#             shift=config["sample_shift"],
#             sampling_steps=config["sample_steps"],
#             text_guide_scale=config["sample_text_guide_scale"],
#             audio_guide_scale=config["sample_audio_guide_scale"],
#             seed=config["base_seed"],
#             offload_model=config["offload_model"],
#             max_frames_num=config["frame_num"] if config["mode"] == 'clip' else 1000,
#             color_correction_strength=config["color_correction_strength"],
#             extra_args=config,
#         )

#         # Save video file
#         formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
#         formatted_prompt = input_data['prompt'].replace(" ", "_").replace("/", "_")[:50]
#         save_file = f"{config['task']}_{config['size']}_{formatted_prompt}_{formatted_time}"
        
#         output_path = f"{save_file}.mp4"
#         logging.info(f"Saving generated video to {output_path}")
#         save_video_ffmpeg(video, save_file, [input_data['video_audio']], high_quality_save=False)
        
#         return output_path
        
#     except Exception as e:
#         logging.error(f"Video generation with TTS failed: {str(e)}")
#         raise

# @app.route('/generate/with-audio', methods=['POST'])
# def generate_with_audio_endpoint():
#     """Endpoint for generating video with local audio files"""
#     try:
#         input_data = request.get_json()
        
#         if not input_data:
#             return jsonify({"error": "No JSON data provided"}), 400
        
#         # Validate required fields for audio files
#         required_fields = ['prompt', 'cond_image', 'cond_audio']
#         for field in required_fields:
#             if field not in input_data:
#                 return jsonify({"error": f"Missing required field: {field}"}), 400
        
#         job_id = str(uuid.uuid4())
#         logging.info(f"Starting video generation with local audio - Job: {job_id}")
        
#         # Generate video with local audio files
#         output_file = generate_video_with_audio_files(input_data, CONFIG, job_id)
        
#         if output_file and os.path.exists(output_file):
#             return jsonify({
#                 "status": "success",
#                 "job_id": job_id,
#                 "output_file": output_file,
#                 "download_url": f"/download/{os.path.basename(output_file)}"
#             }), 200
#         else:
#             return jsonify({
#                 "status": "error",
#                 "message": "Video generation failed - output file not created"
#             }), 500
            
#     except Exception as e:
#         logging.error(f"Error in video generation with audio: {str(e)}")
#         return jsonify({
#             "status": "error",
#             "message": str(e)
#         }), 500

# @app.route('/generate/with-tts', methods=['POST'])
# def generate_with_tts_endpoint():
#     """Endpoint for generating video with TTS"""
#     try:
#         input_data = request.get_json()
        
#         if not input_data:
#             return jsonify({"error": "No JSON data provided"}), 400
        
#         # Validate required fields for TTS
#         required_fields = ['prompt', 'cond_image', 'tts_audio']
#         for field in required_fields:
#             if field not in input_data:
#                 return jsonify({"error": f"Missing required field: {field}"}), 400
        
#         job_id = str(uuid.uuid4())
#         logging.info(f"Starting video generation with TTS - Job: {job_id}")
        
#         # Generate video with TTS
#         output_file = generate_video_with_tts(input_data, CONFIG, job_id)
        
#         if output_file and os.path.exists(output_file):
#             return jsonify({
#                 "status": "success",
#                 "job_id": job_id,
#                 "output_file": output_file,
#                 "download_url": f"/download/{os.path.basename(output_file)}"
#             }), 200
#         else:
#             return jsonify({
#                 "status": "error",
#                 "message": "Video generation failed - output file not created"
#             }), 500
            
#     except Exception as e:
#         logging.error(f"Error in video generation with TTS: {str(e)}")
#         return jsonify({
#             "status": "error",
#             "message": str(e)
#         }), 500

# @app.route('/download/<filename>', methods=['GET'])
# def download_video(filename):
#     try:
#         if os.path.exists(filename):
#             return send_file(filename, as_attachment=True)
#         else:
#             matches = glob.glob(f"*{filename}*")
#             if matches:
#                 return send_file(matches[0], as_attachment=True)
#             else:
#                 return jsonify({"error": "File not found"}), 404
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route('/health', methods=['GET'])
# def health_check():
#     status = "healthy" if pipeline_initialized else "models_not_loaded"
#     return jsonify({
#         "status": status,
#         "models_initialized": pipeline_initialized
#     }), 200

# @app.route('/config', methods=['GET'])
# def get_config():
#     """Endpoint to get current configuration"""
#     return jsonify({
#         "config": CONFIG,
#         "models_initialized": pipeline_initialized
#     }), 200

# def cleanup():
#     """Cleanup function to properly shutdown distributed process group"""
#     if dist.is_initialized():
#         dist.destroy_process_group()

# if __name__ == "__main__":
#     # Set up logging
#     logging.basicConfig(
#         level=logging.INFO,
#         format="[%(asctime)s] %(levelname)s: %(message)s",
#         handlers=[logging.StreamHandler(stream=sys.stdout)]
#     )
    
#     # Install localtunnel if not available
#     try:
#         import subprocess
#         result = subprocess.run(["which", "lt"], capture_output=True, text=True)
#         if result.returncode != 0:
#             logging.info("Installing localtunnel...")
#             subprocess.run(["npm", "install", "-g", "localtunnel"], check=True)
#             logging.info("✅ LocalTunnel installed successfully")
#     except Exception as e:
#         logging.warning(f"Could not install localtunnel: {e}")
    
#     # Validate configuration
#     try:
#         _validate_config(CONFIG)
#         logging.info("Configuration validated successfully")
#     except Exception as e:
#         logging.error(f"Configuration error: {e}")
#         sys.exit(1)
    
#     # Initialize models at startup
#     if initialize_models(CONFIG):
#         logging.info("🎉 ALL MODELS LOADED SUCCESSFULLY!")
#         logging.info("🚀 Flask server starting with pre-loaded models...")
        
#         local_url = "http://localhost:5000"
        
#         logging.info("")
#         logging.info("📋 LOCAL ENDPOINTS:")
#         logging.info(f"   Health Check: GET {local_url}/health")
#         logging.info(f"   TTS Generation: POST {local_url}/generate/with-tts") 
#         logging.info(f"   Audio Generation: POST {local_url}/generate/with-audio")
#         logging.info(f"   Config: GET {local_url}/config")
#         logging.info("")
#         logging.info("🌐 TO GET PUBLIC URL FOR POSTMAN:")
#         logging.info("   1. Keep this server running")
#         logging.info("   2. Open a NEW Colab notebook")
#         logging.info("   3. Run: !lt --port 5000")
#         logging.info("   4. Copy the URL it gives you (looks like: https://abc123.loca.lt)")
#         logging.info("   5. Use that URL in Postman")
#         logging.info("")
#         logging.info("⏳ Starting server on port 5000...")
        
#         try:
#             app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)
#         finally:
#             cleanup()
#     else:
#         logging.error("Failed to initialize models. Server cannot start.")
#         sys.exit(1)





# app.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import uuid
import threading
from datetime import datetime
import subprocess
import logging
from pyngrok import ngrok

# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image
import subprocess

import wan
from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.utils import cache_image, cache_video, str2bool
from wan.utils.multitalk_utils import save_video_ffmpeg
from kokoro import KPipeline
from transformers import Wav2Vec2FeatureExtractor
from src.audio_analysis.wav2vec2 import Wav2Vec2Model

import librosa
import pyloudnorm as pyln
import numpy as np
from einops import rearrange
import soundfile as sf
import re

# Flask app setup
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Global variables for model
wan_pipeline = None
wav2vec_feature_extractor = None
audio_encoder = None
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def custom_init(device, wav2vec_dir):    
    audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec_dir, local_files_only=True).to(device)
    audio_encoder.feature_extractor._freeze_parameters()
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_dir, local_files_only=True)
    return wav2vec_feature_extractor, audio_encoder

def loudness_norm(audio_array, sr=16000, lufs=-23):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
    return normalized_audio

def get_embedding(speech_array, wav2vec_feature_extractor, audio_encoder, sr=16000, device='cpu'):
    audio_duration = len(speech_array) / sr
    video_length = audio_duration * 25  # Assume the video fps is 25

    # wav2vec_feature_extractor
    audio_feature = np.squeeze(
        wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
    )
    audio_feature = torch.from_numpy(audio_feature).float().to(device=device)
    audio_feature = audio_feature.unsqueeze(0)

    # audio encoder
    with torch.no_grad():
        embeddings = audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)

    if len(embeddings) == 0:
        print("Fail to extract audio embedding")
        return None

    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    audio_emb = rearrange(audio_emb, "b s d -> s b d")

    audio_emb = audio_emb.cpu().detach()
    return audio_emb

def extract_audio_from_video(filename, sample_rate):
    raw_audio_path = filename.split('/')[-1].split('.')[0] + '.wav'
    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-i",
        str(filename),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "2",
        str(raw_audio_path),
    ]
    subprocess.run(ffmpeg_command, check=True)
    human_speech_array, sr = librosa.load(raw_audio_path, sr=sample_rate)
    human_speech_array = loudness_norm(human_speech_array, sr)
    os.remove(raw_audio_path)

    return human_speech_array

def audio_prepare_single(audio_path, sample_rate=16000):
    ext = os.path.splitext(audio_path)[1].lower()
    if ext in ['.mp4', '.mov', '.avi', '.mkv']:
        human_speech_array = extract_audio_from_video(audio_path, sample_rate)
        return human_speech_array
    else:
        human_speech_array, sr = librosa.load(audio_path, sr=sample_rate)
        human_speech_array = loudness_norm(human_speech_array, sr)
        return human_speech_array

def process_tts_single(text, save_dir, voice1):    
    s1_sentences = []
    pipeline = KPipeline(lang_code='a', repo_id='/content/Multi-Talk/weights/Kokoro-82M')
    voice_tensor = torch.load(voice1, weights_only=True)
    generator = pipeline(
        text, voice=voice_tensor,
        speed=1, split_pattern=r'\n+'
    )
    audios = []
    for i, (gs, ps, audio) in enumerate(generator):
        audios.append(audio)
    audios = torch.concat(audios, dim=0)
    s1_sentences.append(audios)
    s1_sentences = torch.concat(s1_sentences, dim=0)
    save_path1 = f'{save_dir}/s1.wav'
    sf.write(save_path1, s1_sentences, 24000)
    s1, _ = librosa.load(save_path1, sr=16000)
    return s1, save_path1

def process_tts_multi(text, save_dir, voice1, voice2):
    pattern = r'\(s(\d+)\)\s*(.?)(?=\s\(s\d+\)|$)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    s1_sentences = []
    s2_sentences = []

    pipeline = KPipeline(lang_code='a', repo_id='/content/Multi-Talk/weights/Kokoro-82M')
    for idx, (speaker, content) in enumerate(matches):
        if speaker == '1':
            voice_tensor = torch.load(voice1, weights_only=True)
            generator = pipeline(
                content, voice=voice_tensor,
                speed=1, split_pattern=r'\n+'
            )
            audios = []
            for i, (gs, ps, audio) in enumerate(generator):
                audios.append(audio)
            audios = torch.concat(audios, dim=0)
            s1_sentences.append(audios)
            s2_sentences.append(torch.zeros_like(audios))
        elif speaker == '2':
            voice_tensor = torch.load(voice2, weights_only=True)
            generator = pipeline(
                content, voice=voice_tensor,
                speed=1, split_pattern=r'\n+'
            )
            audios = []
            for i, (gs, ps, audio) in enumerate(generator):
                audios.append(audio)
            audios = torch.concat(audios, dim=0)
            s2_sentences.append(audios)
            s1_sentences.append(torch.zeros_like(audios))
    
    s1_sentences = torch.concat(s1_sentences, dim=0)
    s2_sentences = torch.concat(s2_sentences, dim=0)
    sum_sentences = s1_sentences + s2_sentences
    save_path1 = f'{save_dir}/s1.wav'
    save_path2 = f'{save_dir}/s2.wav'
    save_path_sum = f'{save_dir}/sum.wav'
    sf.write(save_path1, s1_sentences, 24000)
    sf.write(save_path2, s2_sentences, 24000)
    sf.write(save_path_sum, sum_sentences, 24000)

    s1, _ = librosa.load(save_path1, sr=16000)
    s2, _ = librosa.load(save_path2, sr=16000)
    return s1, s2, save_path_sum

def process_tts_triple(text, save_dir, voice1, voice2, voice3):
    print(f"🔍 Processing TTS text: '{text}'")
    
    pattern = r'\(s(\d+)\)\s*([^()]?)(?=\s\(s\d+\)|\s*$)'
    matches = re.findall(pattern, text)
    
    print(f"🔍 Found {len(matches)} speech segments")
    for i, (speaker, content) in enumerate(matches):
        content = content.strip()
        print(f"  Segment {i}: Speaker {speaker}, Content: '{content}'")
    
    if not matches:
        print("❌ No speech segments found! Trying alternative pattern...")
        pattern2 = r'\(s(\d+)\)\s*(.*?)(?=\(s\d+\)|$)'
        matches = re.findall(pattern2, text, re.DOTALL)
        print(f"🔍 Alternative pattern found {len(matches)} segments")
        for i, (speaker, content) in enumerate(matches):
            content = content.strip()
            print(f"  Segment {i}: Speaker {speaker}, Content: '{content}'")
    
    if not matches:
        print("❌ Still no segments found. Creating fallback audio...")
        fallback_audio = np.zeros(16000)
        fallback_path = os.path.join(save_dir, 'sum.wav')
        sf.write(fallback_path, fallback_audio, 16000)
        return fallback_audio, fallback_audio, fallback_audio, fallback_path
    
    s1_sentences = []
    s2_sentences = []
    s3_sentences = []

    pipeline = KPipeline(lang_code='a', repo_id='/content/Multi-Talk/weights/Kokoro-82M')
    
    for idx, (speaker, content) in enumerate(matches):
        content = content.strip()
        if not content:
            print(f"⚠ Skipping empty content for speaker {speaker}")
            continue
            
        print(f"🎤 Processing speaker {speaker}: '{content}'")
        
        try:
            if speaker == '1':
                voice_tensor = torch.load(voice1, weights_only=True)
            elif speaker == '2':
                voice_tensor = torch.load(voice2, weights_only=True)
            elif speaker == '3':
                voice_tensor = torch.load(voice3, weights_only=True)
            else:
                print(f"❌ Unknown speaker: {speaker}")
                continue
                
            generator = pipeline(
                content, voice=voice_tensor,
                speed=1, split_pattern=r'\n+'
            )
            audios = []
            for i, (gs, ps, audio) in enumerate(generator):
                audios.append(audio)
                print(f"  Generated audio chunk {i}: {audio.shape}")
            
            if audios:
                combined_audio = torch.concat(audios, dim=0)
                print(f"✅ Combined audio for speaker {speaker}: {combined_audio.shape}")
                
                if speaker == '1':
                    s1_sentences.append(combined_audio)
                    s2_sentences.append(torch.zeros_like(combined_audio))
                    s3_sentences.append(torch.zeros_like(combined_audio))
                elif speaker == '2':
                    s2_sentences.append(combined_audio)
                    s1_sentences.append(torch.zeros_like(combined_audio))
                    s3_sentences.append(torch.zeros_like(combined_audio))
                elif speaker == '3':
                    s3_sentences.append(combined_audio)
                    s1_sentences.append(torch.zeros_like(combined_audio))
                    s2_sentences.append(torch.zeros_like(combined_audio))
            else:
                print(f"❌ No audio generated for speaker {speaker}")
                
        except Exception as e:
            print(f"❌ Error processing speaker {speaker}: {e}")
            import traceback
            traceback.print_exc()

    print(f"📊 Audio segments - s1: {len(s1_sentences)}, s2: {len(s2_sentences)}, s3: {len(s3_sentences)}")
    
    if not s1_sentences and not s2_sentences and not s3_sentences:
        print("❌ No audio was generated for any speaker! Creating fallback...")
        fallback_audio = np.zeros(16000)
        fallback_path = os.path.join(save_dir, 'sum.wav')
        sf.write(fallback_path, fallback_audio, 16000)
        return fallback_audio, fallback_audio, fallback_audio, fallback_path

    if s1_sentences:
        s1_combined = torch.concat(s1_sentences, dim=0)
        print(f"✅ Final speaker 1 audio: {s1_combined.shape}")
    else:
        s1_combined = torch.tensor([])
        print("⚠ No audio for speaker 1")
        
    if s2_sentences:
        s2_combined = torch.concat(s2_sentences, dim=0)
        print(f"✅ Final speaker 2 audio: {s2_combined.shape}")
    else:
        s2_combined = torch.tensor([])
        print("⚠ No audio for speaker 2")
        
    if s3_sentences:
        s3_combined = torch.concat(s3_sentences, dim=0)
        print(f"✅ Final speaker 3 audio: {s3_combined.shape}")
    else:
        s3_combined = torch.tensor([])
        print("⚠ No audio for speaker 3")

    lengths = [len(s1_combined), len(s2_combined), len(s3_combined)]
    max_length = max(lengths) if lengths else 0
    
    print(f"📏 Audio lengths - s1: {len(s1_combined)}, s2: {len(s2_combined)}, s3: {len(s3_combined)}, max: {max_length}")
    
    if max_length == 0:
        print("❌ All audio is empty! Creating fallback...")
        fallback_audio = np.zeros(16000)
        fallback_path = os.path.join(save_dir, 'sum.wav')
        sf.write(fallback_path, fallback_audio, 16000)
        return fallback_audio, fallback_audio, fallback_audio, fallback_path

    if len(s1_combined) < max_length:
        s1_combined = torch.cat([s1_combined, torch.zeros(max_length - len(s1_combined))])
    if len(s2_combined) < max_length:
        s2_combined = torch.cat([s2_combined, torch.zeros(max_length - len(s2_combined))])
    if len(s3_combined) < max_length:
        s3_combined = torch.cat([s3_combined, torch.zeros(max_length - len(s3_combined))])

    sum_combined = s1_combined + s2_combined + s3_combined
    
    save_path1 = f'{save_dir}/s1.wav'
    save_path2 = f'{save_dir}/s2.wav'
    save_path3 = f'{save_dir}/s3.wav'
    save_path_sum = f'{save_dir}/sum.wav'
    
    if len(s1_combined) > 0:
        sf.write(save_path1, s1_combined.numpy(), 24000)
        print(f"💾 Saved speaker 1 audio: {save_path1}")
    if len(s2_combined) > 0:
        sf.write(save_path2, s2_combined.numpy(), 24000)
        print(f"💾 Saved speaker 2 audio: {save_path2}")
    if len(s3_combined) > 0:
        sf.write(save_path3, s3_combined.numpy(), 24000)
        print(f"💾 Saved speaker 3 audio: {save_path3}")
    
    sf.write(save_path_sum, sum_combined.numpy(), 24000)
    print(f"💾 Saved mixed audio: {save_path_sum}")

    s1, _ = librosa.load(save_path1, sr=16000) if len(s1_combined) > 0 else (np.array([]), 16000)
    s2, _ = librosa.load(save_path2, sr=16000) if len(s2_combined) > 0 else (np.array([]), 16000)
    s3, _ = librosa.load(save_path3, sr=16000) if len(s3_combined) > 0 else (np.array([]), 16000)
    
    print(f"🎉 TTS processing completed successfully!")
    print(f"📈 Final audio lengths - s1: {len(s1)}, s2: {len(s2)}, s3: {len(s3)}")
    
    return s1, s2, s3, save_path_sum

def initialize_models():
    """Initialize the WAN models"""
    global wan_pipeline, wav2vec_feature_extractor, audio_encoder
    
    try:
        logger.info("Initializing WAN models...")
        
        # Initialize Wav2Vec2
        wav2vec_feature_extractor, audio_encoder = custom_init(
            device, 
            '/content/drive/MyDrive/weights/chinese-wav2vec2-base'
        )
        
        # Initialize WAN pipeline
        cfg = WAN_CONFIGS["multitalk-14B"]
        
        wan_pipeline = wan.MultiTalkPipeline(
            config=cfg,
            checkpoint_dir='/content/drive/MyDrive/weights/Wan2.1-I2V-14B-480P',
            quant_dir='/content/drive/MyDrive/weights/MeiGen-MultiTalk',
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            t5_cpu=False,
            lora_dir=['/content/drive/MyDrive/weights/MeiGen-MultiTalk/quant_models/quant_model_int8_FusionX.safetensors'],
            lora_scales=[1.2],
            quant="int8"
        )
        
        logger.info("WAN models initialized successfully!")
        
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        raise

def generate_video_worker(input_data, output_path, job_id):
    """Worker function to generate video in background"""
    try:
        logger.info(f"Starting video generation for job {job_id}")
        
        # Generate video using WAN pipeline
        video = wan_pipeline.generate(
            input_data,
            size_buckget="multitalk-480",
            motion_frame=25,
            frame_num=81,
            shift=2,
            sampling_steps=8,
            text_guide_scale=1.0,
            audio_guide_scale=2.0,
            seed=42,
            offload_model=True,
            max_frames_num=81,
            color_correction_strength=1.0,
            extra_args=None,
        )
        
        # Save video
        video_audio_path = input_data.get('video_audio', '')
        audio_paths = [video_audio_path] if video_audio_path else []
        
        save_video_ffmpeg(video, output_path, audio_paths, high_quality_save=False)
        
        logger.info(f"Video generation completed for job {job_id}")
        
    except Exception as e:
        logger.error(f"Error in video generation for job {job_id}: {e}")

# Flask Routes
@app.route('/')
def home():
    return jsonify({"message": "WAN Video Generation API", "status": "running"})

@app.route('/api/generate-tts-video', methods=['POST'])
def generate_tts_video():
    """Endpoint for TTS-based video generation"""
    try:
        # Get uploaded files and data
        image_file = request.files.get('image')
        config_data = json.loads(request.form.get('config', '{}'))
        
        if not image_file:
            return jsonify({"error": "No image file provided"}), 400
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        job_folder = os.path.join(UPLOAD_FOLDER, job_id)
        os.makedirs(job_folder, exist_ok=True)
        
        # Save uploaded image
        image_path = os.path.join(job_folder, image_file.filename)
        image_file.save(image_path)
        
        # Prepare input data for WAN
        input_data = {
            "prompt": config_data.get("prompt", "A new avatar video."),
            "cond_image": image_path,
            "audio_type": "para",
            "tts_audio": config_data.get("tts_audio", {}),
            "cond_audio": {}
        }
        
        # Log the received data
        logger.info(f"🎯 Received TTS video generation request:")
        logger.info(f"   Job ID: {job_id}")
        logger.info(f"   Image: {image_file.filename}")
        logger.info(f"   Prompt: {input_data['prompt']}")
        logger.info(f"   TTS Audio config: {json.dumps(input_data['tts_audio'], indent=2)}")
        
        # Process TTS audio
        audio_save_dir = os.path.join(job_folder, 'audio')
        os.makedirs(audio_save_dir, exist_ok=True)
        
        tts_audio = input_data['tts_audio']
        text = tts_audio.get('text', '')
        
        logger.info(f"🔊 Processing TTS for text: '{text}'")
        
        if 'human2_voice' not in tts_audio.keys():
            # Single speaker
            logger.info("🎤 Single speaker TTS")
            new_human_speech1, sum_audio = process_tts_single(
                text, audio_save_dir, tts_audio['human1_voice']
            )
            audio_embedding_1 = get_embedding(
                new_human_speech1, wav2vec_feature_extractor, audio_encoder, 16000, device
            )
            emb1_path = os.path.join(audio_save_dir, '1.pt')
            torch.save(audio_embedding_1, emb1_path)
            input_data['cond_audio']['person1'] = emb1_path
            input_data['video_audio'] = sum_audio
            
        elif 'human3_voice' in tts_audio.keys() and tts_audio['human3_voice']:
            # Three speakers
            logger.info("🎤 Three speakers TTS")
            new_human_speech1, new_human_speech2, new_human_speech3, sum_audio = process_tts_triple(
                text, audio_save_dir,
                tts_audio['human1_voice'],
                tts_audio['human2_voice'],
                tts_audio['human3_voice']
            )
            audio_embedding_1 = get_embedding(new_human_speech1, wav2vec_feature_extractor, audio_encoder, 16000, device)
            audio_embedding_2 = get_embedding(new_human_speech2, wav2vec_feature_extractor, audio_encoder, 16000, device)
            audio_embedding_3 = get_embedding(new_human_speech3, wav2vec_feature_extractor, audio_encoder, 16000, device)
            
            emb1_path = os.path.join(audio_save_dir, '1.pt')
            emb2_path = os.path.join(audio_save_dir, '2.pt')
            emb3_path = os.path.join(audio_save_dir, '3.pt')
            
            torch.save(audio_embedding_1, emb1_path)
            torch.save(audio_embedding_2, emb2_path)
            torch.save(audio_embedding_3, emb3_path)
            
            input_data['cond_audio']['person1'] = emb1_path
            input_data['cond_audio']['person2'] = emb2_path
            input_data['cond_audio']['person3'] = emb3_path
            input_data['video_audio'] = sum_audio
        else:
            # Two speakers
            logger.info("🎤 Two speakers TTS")
            new_human_speech1, new_human_speech2, sum_audio = process_tts_multi(
                text, audio_save_dir,
                tts_audio['human1_voice'],
                tts_audio['human2_voice']
            )
            audio_embedding_1 = get_embedding(new_human_speech1, wav2vec_feature_extractor, audio_encoder, 16000, device)
            audio_embedding_2 = get_embedding(new_human_speech2, wav2vec_feature_extractor, audio_encoder, 16000, device)
            
            emb1_path = os.path.join(audio_save_dir, '1.pt')
            emb2_path = os.path.join(audio_save_dir, '2.pt')
            
            torch.save(audio_embedding_1, emb1_path)
            torch.save(audio_embedding_2, emb2_path)
            
            input_data['cond_audio']['person1'] = emb1_path
            input_data['cond_audio']['person2'] = emb2_path
            input_data['video_audio'] = sum_audio
        
        # Log final input data for WAN
        logger.info(f"📋 Final input data for WAN:")
        logger.info(f"   Prompt: {input_data['prompt']}")
        logger.info(f"   Image: {input_data['cond_image']}")
        logger.info(f"   Audio embeddings: {list(input_data['cond_audio'].keys())}")
        logger.info(f"   Video audio: {input_data['video_audio']}")
        
        # Start video generation in background thread
        output_path = os.path.join(OUTPUT_FOLDER, f"video_{job_id}")
        thread = threading.Thread(
            target=generate_video_worker,
            args=(input_data, output_path, job_id)
        )
        thread.start()
        
        return jsonify({
            "job_id": job_id,
            "status": "started",
            "message": "Video generation started"
        })
        
    except Exception as e:
        logger.error(f"❌ Error in TTS video generation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate-audio-video', methods=['POST'])
def generate_audio_video():
    """Endpoint for audio file-based video generation"""
    try:
        # Get uploaded files
        image_file = request.files.get('image')
        audio_files = request.files.getlist('audio_files')
        config_data = json.loads(request.form.get('config', '{}'))
        
        if not image_file or not audio_files:
            return jsonify({"error": "Image and audio files are required"}), 400
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        job_folder = os.path.join(UPLOAD_FOLDER, job_id)
        os.makedirs(job_folder, exist_ok=True)
        
        # Save uploaded image
        image_path = os.path.join(job_folder, image_file.filename)
        image_file.save(image_path)
        
        # Save and process audio files
        audio_save_dir = os.path.join(job_folder, 'audio')
        os.makedirs(audio_save_dir, exist_ok=True)
        
        cond_audio = {}
        all_audio_arrays = []
        all_embeddings = []
        
        for i, audio_file in enumerate(audio_files):
            audio_path = os.path.join(job_folder, f'person{i+1}.wav')
            audio_file.save(audio_path)
            
            # Process audio
            speech = audio_prepare_single(audio_path)
            all_audio_arrays.append(speech)
            
            emb = get_embedding(speech, wav2vec_feature_extractor, audio_encoder, 16000, device)
            emb_path = os.path.join(audio_save_dir, f'{i+1}.pt')
            torch.save(emb, emb_path)
            
            cond_audio[f'person{i+1}'] = emb_path
            all_embeddings.append(emb)
        
        # Create mixed audio
        if all_audio_arrays:
            max_len = max([len(a) for a in all_audio_arrays])
            padded = [np.pad(a, (0, max_len - len(a))) for a in all_audio_arrays]
            sum_audio = np.sum(padded, axis=0)
            sum_audio_path = os.path.join(audio_save_dir, 'sum.wav')
            sf.write(sum_audio_path, sum_audio, 16000)
        
        # Prepare input data
        input_data = {
            "prompt": config_data.get("prompt", "A new avatar video."),
            "cond_image": image_path,
            "audio_type": "para",
            "cond_audio": cond_audio,
            "video_audio": sum_audio_path
        }
        
        # Start video generation in background thread
        output_path = os.path.join(OUTPUT_FOLDER, f"video_{job_id}")
        thread = threading.Thread(
            target=generate_video_worker,
            args=(input_data, output_path, job_id)
        )
        thread.start()
        
        return jsonify({
            "job_id": job_id,
            "status": "started",
            "message": "Audio video generation started"
        })
        
    except Exception as e:
        logger.error(f"Error in audio video generation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """Check status of a generation job"""
    video_path = os.path.join(OUTPUT_FOLDER, f"video_{job_id}.mp4")
    
    if os.path.exists(video_path):
        return jsonify({
            "job_id": job_id,
            "status": "completed",
            "video_url": f"/api/download/{job_id}"
        })
    else:
        return jsonify({
            "job_id": job_id,
            "status": "processing"
        })

@app.route('/api/download/<job_id>', methods=['GET'])
def download_video(job_id):
    """Download generated video"""
    video_path = os.path.join(OUTPUT_FOLDER, f"video_{job_id}.mp4")
    
    if os.path.exists(video_path):
        return send_file(video_path, as_attachment=True)
    else:
        return jsonify({"error": "Video not found"}), 404

def start_ngrok():
    """Start ngrok tunnel"""
    try:
        # Set your ngrok authtoken here
        # You can get it from https://dashboard.ngrok.com/get-started/your-authtoken
        ngrok.set_auth_token("cr_34emEd0Ln96iv1mA71I65fDQYBd")
        
        # Start ngrok tunnel
        public_url = ngrok.connect(5000).public_url
        logger.info(f"Ngrok tunnel created: {public_url}")
        return public_url
    except Exception as e:
        logger.error(f"Error starting ngrok: {e}")
        return None

if __name__ == '__main__':
    # Initialize models
    initialize_models()
    
    # Start ngrok tunnel
    public_url = start_ngrok()
    if public_url:
        logger.info(f"🚀 Server is publicly accessible at: {public_url}")
    
    # Start Flask app
    logger.info("Starting Flask server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)