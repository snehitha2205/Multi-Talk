# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import os
import sys
import json
import warnings
from datetime import datetime
from flask import Flask, request, jsonify, send_file
import uuid
import glob

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

# Import ngrok
from pyngrok import ngrok

app = Flask(__name__)

# Configuration - USING YOUR PATHS FROM CLI
CONFIG = {
    "ckpt_dir": "/content/drive/MyDrive/weights/Wan2.1-I2V-14B-480P",
    "quant_dir": "/content/drive/MyDrive/weights/MeiGen-MultiTalk", 
    "wav2vec_dir": "/content/drive/MyDrive/weights/chinese-wav2vec2-base",
    "task": "multitalk-14B",
    "size": "multitalk-480",
    "frame_num": 81,
    "lora_dir": None,
    "lora_scale": [1.2],
    "offload_model": None,
    "ulysses_size": 1,
    "ring_size": 1,
    "t5_fsdp": False,
    "t5_cpu": False,
    "dit_fsdp": False,
    "audio_save_dir": "save_audio",
    "base_seed": 42,
    "motion_frame": 25,
    "mode": "streaming",
    "sample_steps": 40,
    "sample_shift": None,
    "sample_text_guide_scale": 5.0,
    "sample_audio_guide_scale": 4.0,
    "num_persistent_param_in_dit": None,
    "use_teacache": False,
    "teacache_thresh": 0.2,
    "use_apg": False,
    "apg_momentum": -0.75,
    "apg_norm_threshold": 55,
    "color_correction_strength": 1.0,
    "quant": "int8",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Global variables for pre-loaded models
wan_pipeline = None
wav2vec_feature_extractor = None
audio_encoder = None
pipeline_initialized = False

def _validate_config(config):
    """Validate configuration matching CLI _validate_args functionality"""
    if config["ckpt_dir"] is None:
        raise ValueError("Please specify the checkpoint directory.")
    if config["task"] not in WAN_CONFIGS:
        raise ValueError(f"Unsupport task: {config['task']}")

    if config["sample_steps"] is None:
        config["sample_steps"] = 40

    if config["sample_shift"] is None:
        if config["size"] == 'multitalk-480':
            config["sample_shift"] = 7
        elif config["size"] == 'multitalk-720':
            config["sample_shift"] = 11
        else:
            config["sample_shift"] = 3.0

    config["base_seed"] = config["base_seed"] if config["base_seed"] >= 0 else random.randint(0, 99999999)
    
    if config["size"] not in SUPPORTED_SIZES[config["task"]]:
        raise ValueError(f"Unsupport size {config['size']} for task {config['task']}, supported sizes are: {', '.join(SUPPORTED_SIZES[config['task']])}")

def custom_init(device, wav2vec):    
    audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec, local_files_only=True).to(device)
    audio_encoder.feature_extractor._freeze_parameters()
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec, local_files_only=True)
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
    video_length = audio_duration * 25

    audio_feature = np.squeeze(
        wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
    )
    audio_feature = torch.from_numpy(audio_feature).float().to(device=device)
    audio_feature = audio_feature.unsqueeze(0)

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
    raw_audio_path = filename.split('/')[-1].split('.')[0]+'.wav'
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
    pipeline = KPipeline(lang_code='a', repo_id='/content/drive/MyDrive/weights/Kokoro-82M')

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

    pipeline = KPipeline(lang_code='a', repo_id='/content/drive/MyDrive/weights/Kokoro-82M')
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
    pattern = r'\(s(\d+)\)\s*(.*?)(?=\s*\(s\d+\)|$)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    s1_sentences = []
    s2_sentences = []
    s3_sentences = []

    pipeline = KPipeline(lang_code='a', repo_id='/content/drive/MyDrive/weights/Kokoro-82M')
    
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
            s3_sentences.append(torch.zeros_like(audios))
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
            s3_sentences.append(torch.zeros_like(audios))
        elif speaker == '3':
            voice_tensor = torch.load(voice3, weights_only=True)
            generator = pipeline(
                content, voice=voice_tensor,
                speed=1, split_pattern=r'\n+'
            )
            audios = []
            for i, (gs, ps, audio) in enumerate(generator):
                audios.append(audio)
            audios = torch.concat(audios, dim=0)
            s3_sentences.append(audios)
            s1_sentences.append(torch.zeros_like(audios))
            s2_sentences.append(torch.zeros_like(audios))
    
    s1_sentences = torch.concat(s1_sentences, dim=0)
    s2_sentences = torch.concat(s2_sentences, dim=0)
    s3_sentences = torch.concat(s3_sentences, dim=0)
    sum_sentences = s1_sentences + s2_sentences + s3_sentences
    
    save_path1 = f'{save_dir}/s1.wav'
    save_path2 = f'{save_dir}/s2.wav'
    save_path3 = f'{save_dir}/s3.wav'
    save_path_sum = f'{save_dir}/sum.wav'
    
    sf.write(save_path1, s1_sentences, 24000)
    sf.write(save_path2, s2_sentences, 24000)
    sf.write(save_path3, s3_sentences, 24000)
    sf.write(save_path_sum, sum_sentences, 24000)

    s1, _ = librosa.load(save_path1, sr=16000)
    s2, _ = librosa.load(save_path2, sr=16000)
    s3, _ = librosa.load(save_path3, sr=16000)
    
    return s1, s2, s3, save_path_sum

def initialize_models(config):
    """Initialize models once at server startup"""
    global wan_pipeline, wav2vec_feature_extractor, audio_encoder, pipeline_initialized
    
    if pipeline_initialized:
        return True
        
    try:
        logging.info("Initializing models...")
        
        # Set device - use GPU if available
        device = torch.device(config["device"])
        if device.type == 'cuda':
            torch.cuda.set_device(0)
        
        # Set environment variables for single GPU operation
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        
        # Initialize distributed process group for single GPU
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=0,
                world_size=1
            )
        
        # Initialize Wav2Vec models on correct device
        logging.info("Loading Wav2Vec models...")
        wav2vec_feature_extractor, audio_encoder = custom_init(device, config["wav2vec_dir"])
        
        # Load Wan pipeline configuration
        cfg = WAN_CONFIGS[config["task"]]
        
        # Set offload_model if None (same logic as original)
        if config["offload_model"] is None:
            config["offload_model"] = False  # Single GPU, no offloading needed
            logging.info(f"offload_model is not specified, set to {config['offload_model']}.")
        
        logging.info(f"Creating MultiTalk pipeline for task: {config['task']}")
        wan_pipeline = wan.MultiTalkPipeline(
            config=cfg,
            checkpoint_dir=config["ckpt_dir"],
            quant_dir=config["quant_dir"],
            device_id=0 if device.type == 'cuda' else -1,
            rank=0,
            t5_fsdp=config["t5_fsdp"],
            dit_fsdp=config["dit_fsdp"], 
            use_usp=False,  
            t5_cpu=config["t5_cpu"],
            lora_dir=config["lora_dir"],
            lora_scales=config["lora_scale"],
            quant=config["quant"]
        )

        if config["num_persistent_param_in_dit"] is not None:
            wan_pipeline.vram_management = True
            wan_pipeline.enable_vram_management(
                num_persistent_param_in_dit=config["num_persistent_param_in_dit"]
            )
        
        pipeline_initialized = True
        logging.info("All models initialized successfully!")
        return True
        
    except Exception as e:
        logging.error(f"Failed to initialize models: {str(e)}")
        return False

def generate_video_with_audio_files(input_data, config, job_id):
    """Generate video using local audio files"""
    global wan_pipeline, wav2vec_feature_extractor, audio_encoder
    
    if not pipeline_initialized:
        raise RuntimeError("Models not initialized. Please restart the server.")
    
    try:
        # Process audio data from local files
        audio_save_dir = os.path.join(config["audio_save_dir"], job_id)
        os.makedirs(audio_save_dir, exist_ok=True)
        
        # Audio processing for local files
        num_speakers = len(input_data['cond_audio'])
        all_audio_arrays = []
        all_embeddings = []
        all_audio_paths = []

        device = torch.device(config["device"])
        
        for i in range(num_speakers):
            key = f'person{i+1}'
            audio_path = input_data['cond_audio'].get(key)
            if audio_path is not None:
                speech = audio_prepare_single(audio_path)
                all_audio_arrays.append(speech)
                emb = get_embedding(speech, wav2vec_feature_extractor, audio_encoder, device=device)
                emb_path = os.path.join(audio_save_dir, f'{i+1}.pt')
                torch.save(emb, emb_path)
                input_data['cond_audio'][key] = emb_path
                all_embeddings.append(emb)
                all_audio_paths.append(audio_path)

        if len(all_audio_arrays) > 0:
            max_len = max([len(a) for a in all_audio_arrays])
            padded = [np.pad(a, (0, max_len - len(a))) for a in all_audio_arrays]
            sum_audio = np.sum(padded, axis=0)
            sum_audio_path = os.path.join(audio_save_dir, 'sum.wav')
            sf.write(sum_audio_path, sum_audio, 16000)
            input_data['video_audio'] = sum_audio_path

        # Generate video using pre-loaded pipeline
        logging.info("Generating video with local audio files...")
        video = wan_pipeline.generate(
            input_data,
            size_buckget=config["size"],
            motion_frame=config["motion_frame"],
            frame_num=config["frame_num"],
            shift=config["sample_shift"],
            sampling_steps=config["sample_steps"],
            text_guide_scale=config["sample_text_guide_scale"],
            audio_guide_scale=config["sample_audio_guide_scale"],
            seed=config["base_seed"],
            offload_model=config["offload_model"],
            max_frames_num=config["frame_num"] if config["mode"] == 'clip' else 1000,
            color_correction_strength=config["color_correction_strength"],
            extra_args=config,
        )

        # Save video file
        formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        formatted_prompt = input_data['prompt'].replace(" ", "_").replace("/", "_")[:50]
        save_file = f"{config['task']}_{config['size']}_{formatted_prompt}_{formatted_time}"
        
        output_path = f"{save_file}.mp4"
        logging.info(f"Saving generated video to {output_path}")
        save_video_ffmpeg(video, save_file, [input_data['video_audio']], high_quality_save=False)
        
        return output_path
        
    except Exception as e:
        logging.error(f"Video generation with local audio failed: {str(e)}")
        raise

def generate_video_with_tts(input_data, config, job_id):
    """Generate video using TTS"""
    global wan_pipeline, wav2vec_feature_extractor, audio_encoder
    
    if not pipeline_initialized:
        raise RuntimeError("Models not initialized. Please restart the server.")
    
    try:
        # Process TTS data
        audio_save_dir = os.path.join(config["audio_save_dir"], job_id)
        os.makedirs(audio_save_dir, exist_ok=True)
        
        device = torch.device(config["device"])
        
        # TTS processing
        if 'human2_voice' not in input_data['tts_audio'].keys():
            # Single speaker TTS
            new_human_speech1, sum_audio = process_tts_single(
                input_data['tts_audio']['text'], 
                audio_save_dir, 
                input_data['tts_audio']['human1_voice']
            )
            audio_embedding_1 = get_embedding(new_human_speech1, wav2vec_feature_extractor, audio_encoder, device=device)
            emb1_path = os.path.join(audio_save_dir, '1.pt')
            torch.save(audio_embedding_1, emb1_path)
            input_data['cond_audio']['person1'] = emb1_path
            input_data['video_audio'] = sum_audio
            
        elif 'human3_voice' in input_data['tts_audio'].keys() and input_data['tts_audio']['human3_voice']:
            # Three speaker TTS
            new_human_speech1, new_human_speech2, new_human_speech3, sum_audio = process_tts_triple(
                input_data['tts_audio']['text'], 
                audio_save_dir, 
                input_data['tts_audio']['human1_voice'], 
                input_data['tts_audio']['human2_voice'],
                input_data['tts_audio']['human3_voice']
            )
            audio_embedding_1 = get_embedding(new_human_speech1, wav2vec_feature_extractor, audio_encoder, device=device)
            audio_embedding_2 = get_embedding(new_human_speech2, wav2vec_feature_extractor, audio_encoder, device=device)
            audio_embedding_3 = get_embedding(new_human_speech3, wav2vec_feature_extractor, audio_encoder, device=device)
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
            # Two speaker TTS
            new_human_speech1, new_human_speech2, sum_audio = process_tts_multi(
                input_data['tts_audio']['text'], 
                audio_save_dir, 
                input_data['tts_audio']['human1_voice'], 
                input_data['tts_audio']['human2_voice']
            )
            audio_embedding_1 = get_embedding(new_human_speech1, wav2vec_feature_extractor, audio_encoder, device=device)
            audio_embedding_2 = get_embedding(new_human_speech2, wav2vec_feature_extractor, audio_encoder, device=device)
            emb1_path = os.path.join(audio_save_dir, '1.pt')
            emb2_path = os.path.join(audio_save_dir, '2.pt')
            torch.save(audio_embedding_1, emb1_path)
            torch.save(audio_embedding_2, emb2_path)
            input_data['cond_audio']['person1'] = emb1_path
            input_data['cond_audio']['person2'] = emb2_path
            input_data['video_audio'] = sum_audio

        # Generate video using pre-loaded pipeline
        logging.info("Generating video with TTS...")
        video = wan_pipeline.generate(
            input_data,
            size_buckget=config["size"],
            motion_frame=config["motion_frame"],
            frame_num=config["frame_num"],
            shift=config["sample_shift"],
            sampling_steps=config["sample_steps"],
            text_guide_scale=config["sample_text_guide_scale"],
            audio_guide_scale=config["sample_audio_guide_scale"],
            seed=config["base_seed"],
            offload_model=config["offload_model"],
            max_frames_num=config["frame_num"] if config["mode"] == 'clip' else 1000,
            color_correction_strength=config["color_correction_strength"],
            extra_args=config,
        )

        # Save video file
        formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        formatted_prompt = input_data['prompt'].replace(" ", "_").replace("/", "_")[:50]
        save_file = f"{config['task']}_{config['size']}_{formatted_prompt}_{formatted_time}"
        
        output_path = f"{save_file}.mp4"
        logging.info(f"Saving generated video to {output_path}")
        save_video_ffmpeg(video, save_file, [input_data['video_audio']], high_quality_save=False)
        
        return output_path
        
    except Exception as e:
        logging.error(f"Video generation with TTS failed: {str(e)}")
        raise

@app.route('/generate/with-audio', methods=['POST'])
def generate_with_audio_endpoint():
    """Endpoint for generating video with local audio files"""
    try:
        input_data = request.get_json()
        
        if not input_data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate required fields for audio files
        required_fields = ['prompt', 'cond_image', 'cond_audio']
        for field in required_fields:
            if field not in input_data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        job_id = str(uuid.uuid4())
        logging.info(f"Starting video generation with local audio - Job: {job_id}")
        
        # Generate video with local audio files
        output_file = generate_video_with_audio_files(input_data, CONFIG, job_id)
        
        if output_file and os.path.exists(output_file):
            return jsonify({
                "status": "success",
                "job_id": job_id,
                "output_file": output_file,
                "download_url": f"/download/{os.path.basename(output_file)}"
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": "Video generation failed - output file not created"
            }), 500
            
    except Exception as e:
        logging.error(f"Error in video generation with audio: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/generate/with-tts', methods=['POST'])
def generate_with_tts_endpoint():
    """Endpoint for generating video with TTS"""
    try:
        input_data = request.get_json()
        
        if not input_data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate required fields for TTS
        required_fields = ['prompt', 'cond_image', 'tts_audio']
        for field in required_fields:
            if field not in input_data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        job_id = str(uuid.uuid4())
        logging.info(f"Starting video generation with TTS - Job: {job_id}")
        
        # Generate video with TTS
        output_file = generate_video_with_tts(input_data, CONFIG, job_id)
        
        if output_file and os.path.exists(output_file):
            return jsonify({
                "status": "success",
                "job_id": job_id,
                "output_file": output_file,
                "download_url": f"/download/{os.path.basename(output_file)}"
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": "Video generation failed - output file not created"
            }), 500
            
    except Exception as e:
        logging.error(f"Error in video generation with TTS: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/download/<filename>', methods=['GET'])
def download_video(filename):
    try:
        if os.path.exists(filename):
            return send_file(filename, as_attachment=True)
        else:
            matches = glob.glob(f"*{filename}*")
            if matches:
                return send_file(matches[0], as_attachment=True)
            else:
                return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    status = "healthy" if pipeline_initialized else "models_not_loaded"
    return jsonify({
        "status": status,
        "models_initialized": pipeline_initialized
    }), 200

@app.route('/config', methods=['GET'])
def get_config():
    """Endpoint to get current configuration"""
    return jsonify({
        "config": CONFIG,
        "models_initialized": pipeline_initialized
    }), 200

def cleanup():
    """Cleanup function to properly shutdown distributed process group"""
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)]
    )
    
    # Validate configuration
    try:
        _validate_config(CONFIG)
        logging.info("Configuration validated successfully")
    except Exception as e:
        logging.error(f"Configuration error: {e}")
        sys.exit(1)
    
    # Initialize models at startup
    if initialize_models(CONFIG):
        logging.info("Flask server starting with pre-loaded models...")
        
        # Start ngrok tunnel
        public_url = ngrok.connect(5000)
        logging.info(f"🚀 Public URL created: {public_url}")
        logging.info("📋 Use this URL in Postman to test the API:")
        logging.info(f"   Health Check: GET {public_url}/health")
        logging.info(f"   TTS Generation: POST {public_url}/generate/with-tts")
        logging.info(f"   Audio Generation: POST {public_url}/generate/with-audio")
        
        try:
            app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)
        finally:
            cleanup()
            ngrok.kill()  # Close ngrok tunnel when server stops
    else:
        logging.error("Failed to initialize models. Server cannot start.")
        sys.exit(1)