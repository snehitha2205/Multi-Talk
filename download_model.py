from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="TencentGameMate/chinese-wav2vec2-base",
    local_dir="./weights/chinese-wav2vec2-base",
    local_dir_use_symlinks=False
)
