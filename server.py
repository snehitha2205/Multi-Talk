# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import os
import sys
import json
import warnings
from datetime import datetime
from flask import Flask, request, jsonify, send_file
import uuid

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

app = Flask(__name__)

# Configuration - you can set these as environment variables or modify directly
CONFIG = {
    "ckpt_dir": None,  # Set your checkpoint directory
    "quant_dir": None,  # Set your quant directory
    "wav2vec_dir": None,  # Set your wav2vec directory
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
    "mode": "clip",
    "sample_steps": None,
    "sample_shift": None,
    "sample_text_guide_scale": 5.0,
    "sample_audio_guide_scale": 4.0,
    "num_persistent_param_in_dit": None,
    "audio_mode": "localfile",
    "use_teacache": False,
    "teacache_thresh": 0.2,
    "use_apg": False,
    "apg_momentum": -0.75,
    "apg_norm_threshold": 55,
    "color_correction_strength": 1.0,
    "quant": None
}

def _validate_config(config):
    # Basic check
    assert config["ckpt_dir"] is not None, "Please specify the checkpoint directory."
    assert config["task"] in WAN_CONFIGS, f"Unsupport task: {config['task']}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
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
    
    # Size check
    assert config["size"] in SUPPORTED_SIZES[config["task"]], f"Unsupport size {config['size']} for task {config['task']}, supported sizes are: {', '.join(SUPPORTED_SIZES[config['task']])}"

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

def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)

def get_embedding(speech_array, wav2vec_feature_extractor, audio_encoder, sr=16000, device='cpu'):
    audio_duration = len(speech_array) / sr
    video_length = audio_duration * 25 # Assume the video fps is 25

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
        text, voice=voice_tensor, # <= change voice here
        speed=1, split_pattern=r'\n+'
    )
    audios = []
    for i, (gs, ps, audio) in enumerate(generator):
        audios.append(audio)
    audios = torch.concat(audios, dim=0)
    s1_sentences.append(audios)
    s1_sentences = torch.concat(s1_sentences, dim=0)
    save_path1 =f'{save_dir}/s1.wav'
    sf.write(save_path1, s1_sentences, 24000) # save each audio file
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
                content, voice=voice_tensor, # <= change voice here
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
                content, voice=voice_tensor, # <= change voice here
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
    save_path1 =f'{save_dir}/s1.wav'
    save_path2 =f'{save_dir}/s2.wav'
    save_path_sum = f'{save_dir}/sum.wav'
    sf.write(save_path1, s1_sentences, 24000) # save each audio file
    sf.write(save_path2, s2_sentences, 24000)
    sf.write(save_path_sum, sum_sentences, 24000)

    s1, _ = librosa.load(save_path1, sr=16000)
    s2, _ = librosa.load(save_path2, sr=16000)
    # sum, _ = librosa.load(save_path_sum, sr=16000)
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

def generate_video(input_data, config, job_id):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if config["offload_model"] is None:
        config["offload_model"] = False if world_size > 1 else True
        logging.info(f"offload_model is not specified, set to {config['offload_model']}.")
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (config["t5_fsdp"] or config["dit_fsdp"]), "t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (config["ulysses_size"] > 1 or config["ring_size"] > 1), "context parallel are not supported in non-distributed environments."

    if config["ulysses_size"] > 1 or config["ring_size"] > 1:
        assert config["ulysses_size"] * config["ring_size"] == world_size, "The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=config["ring_size"],
            ulysses_degree=config["ulysses_size"],
        )

    cfg = WAN_CONFIGS[config["task"]]
    if config["ulysses_size"] > 1:
        assert cfg.num_heads % config["ulysses_size"] == 0, f"{cfg.num_heads=} cannot be divided evenly by {config['ulysses_size']=}."

    logging.info(f"Generation job config: {config}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [config["base_seed"]] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        config["base_seed"] = base_seed[0]

    if config["task"] not in ["multitalk-14B", "t2v-1.3B"]:
        raise ValueError(f"Unsupported task: {config['task']}")

    # Process audio data
    wav2vec_feature_extractor, audio_encoder = custom_init('cpu', config["wav2vec_dir"])
    config["audio_save_dir"] = os.path.join(config["audio_save_dir"], job_id)
    os.makedirs(config["audio_save_dir"], exist_ok=True)
    
    if config["audio_mode"] == 'localfile':
        num_speakers = len(input_data['cond_audio'])
        all_audio_arrays = []
        all_embeddings = []
        all_audio_paths = []

        for i in range(num_speakers):
            key = f'person{i+1}'
            audio_path = input_data['cond_audio'].get(key)
            if audio_path is not None:
                speech = audio_prepare_single(audio_path)
                all_audio_arrays.append(speech)
                emb = get_embedding(speech, wav2vec_feature_extractor, audio_encoder)
                emb_path = os.path.join(config["audio_save_dir"], f'{i+1}.pt')
                torch.save(emb, emb_path)
                input_data['cond_audio'][key] = emb_path
                all_embeddings.append(emb)
                all_audio_paths.append(audio_path)

        if len(all_audio_arrays) > 0:
            max_len = max([len(a) for a in all_audio_arrays])
            padded = [np.pad(a, (0, max_len - len(a))) for a in all_audio_arrays]
            sum_audio = np.sum(padded, axis=0)
            sum_audio_path = os.path.join(config["audio_save_dir"], 'sum.wav')
            sf.write(sum_audio_path, sum_audio, 16000)
            input_data['video_audio'] = sum_audio_path
    elif config["audio_mode"] == 'tts':
        if 'human2_voice' not in input_data['tts_audio'].keys():
            # Single speaker TTS
            new_human_speech1, sum_audio = process_tts_single(input_data['tts_audio']['text'], config["audio_save_dir"], input_data['tts_audio']['human1_voice'])
            audio_embedding_1 = get_embedding(new_human_speech1, wav2vec_feature_extractor, audio_encoder)
            emb1_path = os.path.join(config["audio_save_dir"], '1.pt')
            torch.save(audio_embedding_1, emb1_path)
            input_data['cond_audio']['person1'] = emb1_path
            input_data['video_audio'] = sum_audio
        elif 'human3_voice' in input_data['tts_audio'].keys() and input_data['tts_audio']['human3_voice']:
            # Three speaker TTS
            new_human_speech1, new_human_speech2, new_human_speech3, sum_audio = process_tts_triple(
                input_data['tts_audio']['text'], 
                config["audio_save_dir"], 
                input_data['tts_audio']['human1_voice'], 
                input_data['tts_audio']['human2_voice'],
                input_data['tts_audio']['human3_voice']
            )
            audio_embedding_1 = get_embedding(new_human_speech1, wav2vec_feature_extractor, audio_encoder)
            audio_embedding_2 = get_embedding(new_human_speech2, wav2vec_feature_extractor, audio_encoder)
            audio_embedding_3 = get_embedding(new_human_speech3, wav2vec_feature_extractor, audio_encoder)
            emb1_path = os.path.join(config["audio_save_dir"], '1.pt')
            emb2_path = os.path.join(config["audio_save_dir"], '2.pt')
            emb3_path = os.path.join(config["audio_save_dir"], '3.pt')
            torch.save(audio_embedding_1, emb1_path)
            torch.save(audio_embedding_2, emb2_path)
            torch.save(audio_embedding_3, emb3_path)
            input_data['cond_audio']['person1'] = emb1_path
            input_data['cond_audio']['person2'] = emb2_path
            input_data['cond_audio']['person3'] = emb3_path
            input_data['video_audio'] = sum_audio
        else:
            # Two speaker TTS
            new_human_speech1, new_human_speech2, sum_audio = process_tts_multi(input_data['tts_audio']['text'], config["audio_save_dir"], input_data['tts_audio']['human1_voice'], input_data['tts_audio']['human2_voice'])
            audio_embedding_1 = get_embedding(new_human_speech1, wav2vec_feature_extractor, audio_encoder)
            audio_embedding_2 = get_embedding(new_human_speech2, wav2vec_feature_extractor, audio_encoder)
            emb1_path = os.path.join(config["audio_save_dir"], '1.pt')
            emb2_path = os.path.join(config["audio_save_dir"], '2.pt')
            torch.save(audio_embedding_1, emb1_path)
            torch.save(audio_embedding_2, emb2_path)
            input_data['cond_audio']['person1'] = emb1_path
            input_data['cond_audio']['person2'] = emb2_path
            input_data['video_audio'] = sum_audio

    logging.info("Creating MultiTalk pipeline.")
    wan_i2v = wan.MultiTalkPipeline(
        config=cfg,
        checkpoint_dir=config["ckpt_dir"],
        quant_dir=config["quant_dir"],
        device_id=device,
        rank=rank,
        t5_fsdp=config["t5_fsdp"],
        dit_fsdp=config["dit_fsdp"], 
        use_usp=(config["ulysses_size"] > 1 or config["ring_size"] > 1),  
        t5_cpu=config["t5_cpu"],
        lora_dir=config["lora_dir"],
        lora_scales=config["lora_scale"],
        quant=config["quant"]
    )

    if config["num_persistent_param_in_dit"] is not None:
        wan_i2v.vram_management = True
        wan_i2v.enable_vram_management(
            num_persistent_param_in_dit=config["num_persistent_param_in_dit"]
        )
    
    logging.info("Generating video ...")
    video = wan_i2v.generate(
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

    if rank == 0:
        save_file = f"output_{job_id}"
        
        if config.get("save_file") is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = input_data['prompt'].replace(" ", "_").replace("/", "_")[:50]
            save_file = f"{config['task']}{config['size'].replace('*','x') if sys.platform=='win32' else config['size']}{config['ulysses_size']}{config['ring_size']}{formatted_prompt}_{formatted_time}"
        else:
            save_file = config["save_file"]
        
        logging.info(f"Saving generated video to {save_file}.mp4")
        save_video_ffmpeg(video, save_file, [input_data['video_audio']], high_quality_save=False)
        
        return f"{save_file}.mp4"
    
    logging.info("Finished.")
    return None

@app.route('/generate', methods=['POST'])
def generate_endpoint():
    try:
        # Get JSON data from request
        input_data = request.get_json()
        
        if not input_data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Validate configuration
        _validate_config(CONFIG)
        
        # Generate video
        output_file = generate_video(input_data, CONFIG, job_id)
        
        if output_file and os.path.exists(output_file):
            return jsonify({
                "status": "success",
                "job_id": job_id,
                "output_file": output_file,
                "download_url": f"/download/{job_id}"
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": "Video generation failed"
            }), 500
            
    except Exception as e:
        logging.error(f"Error in video generation: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/download/<job_id>', methods=['GET'])
def download_video(job_id):
    try:
        # Look for the output file - you might need to adjust this logic based on your naming convention
        possible_files = [
            f"output_{job_id}.mp4",
            f"{CONFIG['task']}*{job_id}*.mp4"
        ]
        
        output_file = None
        for file_pattern in possible_files:
            import glob
            matches = glob.glob(file_pattern)
            if matches:
                output_file = matches[0]
                break
        
        if output_file and os.path.exists(output_file):
            return send_file(output_file, as_attachment=True)
        else:
            return jsonify({"error": "File not found"}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    # Validate configuration on startup
    try:
        _validate_config(CONFIG)
        print("Configuration validated successfully")
    except Exception as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    
    app.run(host='0.0.0.0', port=5000, debug=False)