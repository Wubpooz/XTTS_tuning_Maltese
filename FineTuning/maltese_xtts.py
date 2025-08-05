# -*- coding: utf-8 -*-
"""maltese_xtts.py
This script is used to fine-tune the XTTS model for Maltese language and run inference on it.
It includes functions for training the model and performing inference, as well as a command-line interface to
control the process.
"""

# Inference imports
import torch
import torchaudio
from tqdm import tqdm

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Training imports
import os
import gc
from pathlib import Path

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager

# import shutil
import argparse


# ========================== Training ==========================
def train_gpt(metadatas, num_epochs=100, batch_size=3, grad_acumm=84, output_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints"), lr=5e-06, weight_decay=1e-2, save_step=10000, custom_model="", version="main", max_text_length=200, max_audio_length=255995, multi_gpu=False):
  """Train the GPT XTTS model for Maltese language.
  This function sets up the training configuration, downloads necessary files, initializes the model, and starts the training process.
  It also saves the final model checkpoint and configuration files after training.

  Args:
      metadatas (list): A list of metadata strings in the format "train_csv_path,eval_csv_path,language".
      num_epochs (int): Number of epochs for training. Default is 100.
      batch_size (int): Mini batch size. Default is 3.
      grad_acumm (int): Gradient accumulation steps. Default is 84.
      output_path (str): Path to save the model checkpoints and outputs. Default is the current directory/"checkpoints".
      lr (float): Learning rate for the optimizer. Default is 5e-6.
      weight_decay (float): Weight decay for the optimizer. Default is 1e-2.
      save_step (int): Step interval for saving the model checkpoints. Default is 10000.
      custom_model (str): Path to a custom model checkpoint (.pth file) to use instead of the default XTTS model.
      version (str): XTTS version to use (default: "main"). 
      max_text_length (int): Maximum text length for the model. Default is 200.
      max_audio_length (int): Maximum audio length for the model. Default is 255995 (approximately 12 seconds at 22050 Hz).
      multi_gpu (bool): Whether to use multi-GPU training. Default is False.
  Returns:
      tuple: Paths to the XTTS checkpoint, tokenizer file, config file, trainer output path, and speaker reference audio file.
  """  

  RUN_NAME = "GPT_XTTS_FT"
  PROJECT_NAME = "XTTS_trainer_maltese"
  DASHBOARD_LOGGER = "tensorboard"
  LOGGER_URI = None
  num_workers = 8

  OUT_PATH = os.path.join(output_path, "run", "training") #Path.cwd()
  os.makedirs(OUT_PATH, exist_ok=True)

  # Training Parameters
  OPTIMIZER_WD_ONLY_ON_WEIGHTS = not multi_gpu
  START_WITH_EVAL = False #TODO
  BATCH_SIZE = batch_size
  GRAD_ACUMM_STEPS = grad_acumm

  DATASETS_CONFIG_LIST = []
  for metadata in metadatas:
    train_csv, eval_csv, language = metadata.split(",")
    print(train_csv, eval_csv, language)
    if language == "ja":
      num_workers = 0

    config_dataset = BaseDatasetConfig(
      formatter="coqui",
      dataset_name="ft_dataset",
      path=os.path.dirname(train_csv), #os.path.join(output_path, "dataset")
      meta_file_train=os.path.basename(train_csv),
      meta_file_val=os.path.basename(eval_csv),
      language=language,
    )
    DATASETS_CONFIG_LIST.append(config_dataset)

  CHECKPOINTS_OUT_PATH = os.path.join(output_path, "models", f"{version}")
  os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)


  # DVAE files
  DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
  MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"
  DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
  MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))

  if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
    print(" > Downloading DVAE files!")
    ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)



  TOKENIZER_FILE_LINK = f"https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/{version}/vocab.json"
  XTTS_CHECKPOINT_LINK = f"https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/{version}/model.pth"
  # if useful, can add config and speakers files
  # XTTS_CONFIG_LINK = f"https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/{version}/config.json"
  # XTTS_SPEAKER_LINK = f"https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/speakers_xtts.pth"

  XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))
  TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))

  if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
    print(f" > Downloading XTTS v{version} files!")
    ModelManager._download_model_files([TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True) # private API

  # if useful, can transfer files to ready folder
  # READY_MODEL_PATH = os.path.join(output_path,"ready")
  # if not os.path.exists(READY_MODEL_PATH):
  #   os.makedirs(READY_MODEL_PATH)
  # NEW_TOKENIZER_FILE = os.path.join(READY_MODEL_PATH, "vocab.json")
  # shutil.copy(TOKENIZER_FILE, NEW_TOKENIZER_FILE)
  # TOKENIZER_FILE = NEW_TOKENIZER_FILE # vocab.json file

  if custom_model:
    if os.path.isfile(custom_model) and custom_model.endswith('.pth'):
        XTTS_CHECKPOINT = custom_model
        print(f" > Loading custom model: {XTTS_CHECKPOINT}")
    else:
        raise ValueError(f"Error: The specified custom model is not a valid .pth file: {custom_model}")

  model_args = GPTArgs(
    max_conditioning_length=132300,  # 6 secs
    min_conditioning_length=66150,  # 3 secs   or 11025 for 0.5sec
    debug_loading_failures=False,
    max_wav_length=max_audio_length,
    max_text_length=max_text_length,
    mel_norm_file=MEL_NORM_FILE,
    dvae_checkpoint=DVAE_CHECKPOINT,
    xtts_checkpoint=XTTS_CHECKPOINT,
    tokenizer_file=TOKENIZER_FILE,
    gpt_num_audio_tokens=1026,
    gpt_start_audio_token=1024,
    gpt_stop_audio_token=1025,
    gpt_use_masking_gt_prompt_approach=True,
    gpt_use_perceiver_resampler=True
  )

  audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)

  config = GPTTrainerConfig(
    output_path=OUT_PATH,
    model_args=model_args,
    run_name=RUN_NAME,
    project_name=PROJECT_NAME,
    run_description="""GPT XTTS fine-tuning for Maltese""",
    dashboard_logger=DASHBOARD_LOGGER,
    logger_uri=LOGGER_URI, # type: ignore
    audio=audio_config,
    epochs=num_epochs,
    batch_size=BATCH_SIZE,
    batch_group_size=48,
    eval_batch_size=BATCH_SIZE,
    num_loader_workers=num_workers,
    eval_split_max_size=256,
    print_step=50,
    plot_step=100,
    log_model_step=100,
    save_step=save_step,
    save_n_checkpoints=1,
    save_checkpoints=True,
    # target_loss="loss",
    print_eval=False,
    optimizer="AdamW",
    optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
    optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": weight_decay},
    lr=lr,
    lr_scheduler="MultiStepLR", #TODO use CosineAnnealingLR?
    lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
    test_sentences=[],
  )

  model = GPTTrainer.init_from_config(config)

  train_samples, eval_samples = load_tts_samples(
    DATASETS_CONFIG_LIST,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
  )


  trainer = Trainer(
    TrainerArgs(
      restore_path=None, #type: ignore  # xtts checkpoint is restored via xtts_checkpoint key so no need of restore it using Trainer restore_path parameter
      skip_train_epoch=False,
      start_with_eval=START_WITH_EVAL,
      grad_accum_steps=GRAD_ACUMM_STEPS,
    ),
    config,
    output_path=OUT_PATH, #os.path.join(output_path, "run", "training")
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
  )
  trainer.fit()
  print("Training finished!")

  trainer.save_checkpoint(
    os.path.join(OUT_PATH, "final_model.pth"),
    config=config,
    tokenizer_file=TOKENIZER_FILE,
    dvae_checkpoint=DVAE_CHECKPOINT,
    xtts_checkpoint=XTTS_CHECKPOINT,
  )

  CONFIG_PATH = os.path.join(OUT_PATH, "config.json")
  inference_config = XttsConfig()
  inference_config.model_args = config.model_args  # Copy model args from training
  inference_config.audio = config.audio
  inference_config.save_json(CONFIG_PATH)

  # get the longest text audio file to use as speaker reference
  samples_len = [len(item["text"].split(" ")) for item in train_samples] # type: ignore
  longest_text_idx =  samples_len.index(max(samples_len))
  speaker_ref = train_samples[longest_text_idx]["audio_file"] # type: ignore
  if not os.path.isabs(speaker_ref):
    speaker_ref = os.path.join(output_path, os.path.dirname(metadatas[0].split(",")[0]), speaker_ref)

  trainer_out_path = trainer.output_path

  # deallocate VRAM and RAM
  del model, trainer, train_samples, eval_samples
  gc.collect()

  return XTTS_CHECKPOINT, TOKENIZER_FILE, CONFIG_PATH, trainer_out_path, speaker_ref






# ========================== Inference ==========================
def inference(xtts_checkpoint, xtts_config, xtts_vocab, tts_text, speaker_audio_file, lang):
  """Run inference using the XTTS model with the provided configuration and text.
  Args:
      xtts_checkpoint (str): Path to the XTTS model checkpoint.
      xtts_config (str): Path to the XTTS configuration file.
      xtts_vocab (str): Path to the XTTS vocabulary file.
      tts_text (str): Text to be synthesized.
      speaker_audio_file (str): Path to the audio file of the speaker for conditioning.
      lang (str): Language code for the text. Supported languages include "en", "fr", "de", "es", "it", "pt", "ru", "zh", "ja", "ko", and "mt".
  Returns:
      torch.Tensor: Synthesized audio waveform.
  """

  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  try:
    import deepspeed
    use_deepspeed = device.startswith("cuda")  # Use deepspeed only if CUDA is available
  except ImportError:
    use_deepspeed = False
    print("Deepspeed is not installed, using CPU/GPU without deepspeed.")

  config = XttsConfig()
  print("Loading config...")
  config.load_json(xtts_config)
  print("Config Loaded.")
  print("Initing model...")
  XTTS_MODEL = Xtts.init_from_config(config)
  print("Model Init, loadign checkpoint...")
  XTTS_MODEL.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=use_deepspeed)
  XTTS_MODEL.to(device)
  print("Model loaded successfully!")


  gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
    audio_path=speaker_audio_file,
    gpt_cond_len=XTTS_MODEL.config.gpt_cond_len, # type: ignore
    max_ref_length=XTTS_MODEL.config.max_ref_len, # type: ignore
    sound_norm_refs=XTTS_MODEL.config.sound_norm_refs, # type: ignore
  )

  import nltk
  nltk.download('punkt') # Run once
  from nltk.tokenize import sent_tokenize

  tts_texts = sent_tokenize(tts_text)

  wav_chunks = []
  print("Infering...")
  for text in tqdm(tts_texts):
    wav_chunk = XTTS_MODEL.inference(
      text=text,
      language=lang,
      gpt_cond_latent=gpt_cond_latent,
      speaker_embedding=speaker_embedding,
      temperature=0.1,
      length_penalty=1.0,
      repetition_penalty=10.0,
      top_k=10,
      top_p=0.3,
    )
    wav_chunks.append(torch.tensor(wav_chunk["wav"]))
  print("Inference successful!")

  return torch.cat(wav_chunks, dim=0).unsqueeze(0).cpu()





# ========================== CLI Parser ==========================
def create_xtts_trainer_parser():
  parser = argparse.ArgumentParser(description="Arguments for XTTS Trainer")
  parser.add_argument("--output_path", type=str, required=True, help="Path to pretrained + checkpoint model")
  parser.add_argument("--metadatas", nargs='+', type=str, required=True, help="train_csv_path,eval_csv_path,language")
  parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
  parser.add_argument("--batch_size", type=int, default=3, help="Mini batch size")
  parser.add_argument("--grad_acumm", type=int, default=84, help="Grad accumulation steps")
  parser.add_argument("--max_audio_length", type=int, default=255995, help="Max audio length")
  parser.add_argument("--max_text_length", type=int, default=200, help="Max text length")
  parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
  parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
  parser.add_argument("--save_step", type=int, default=10000, help="Save step")
  parser.add_argument("--custom_model", type=str, default="", help="Path to custom model checkpoint (.pth file)")
  parser.add_argument("--version", type=str, default="main", help="XTTS version to use (default: main)")
  parser.add_argument("--multi_gpu", action='store_true', help="Use multi-GPU training")
  # parser.add_argument("--no_deepspeed", action='store_true', help="Disable deepspeed for training")
  return parser


if __name__ == "__main__":
  parser = create_xtts_trainer_parser()
  args = parser.parse_args()

  xtts_checkpoint, xtts_vocab, config, trainer_out_path, speaker_ref = train_gpt(
    metadatas=args.metadatas,
    output_path=args.output_path,
    num_epochs=args.num_epochs,
    batch_size=args.batch_size,
    grad_acumm=args.grad_acumm,
    weight_decay=args.weight_decay,
    lr=args.lr,
    max_text_length=args.max_text_length,
    max_audio_length=args.max_audio_length,
    save_step=args.save_step
  )

  print(f"Checkpoint saved in dir: {trainer_out_path}")


  run_inference = input("Do you want to run inference? (y/n): ").strip().lower()
  if run_inference == 'y':
    print("Running inference...")
    audio = inference(
      xtts_checkpoint=xtts_checkpoint,
      xtts_config=config,
      xtts_vocab=xtts_vocab,
      tts_text="Hija test tal-mudell tat-taħdit il-ġdid tiegħi, il-lingwa Maltija hija interessanti! Esperimenti u testijiet huma importanti biex niskopru l-possibbiltajiet tat-taħdit.",
      speaker_audio_file=speaker_ref,
      lang="mt"
    )
    print("Inference completed!")
    torchaudio.save(os.path.join(trainer_out_path, "output_maltese.wav"), audio, 24000)

    try:
      from IPython.display import Audio
      Audio(audio, rate=24000) # Play audio (for Jupyter Notebook)
    except ImportError:
      print("IPython not available, audio playback not supported in this environment.")
  else:
    print("Skipping inference. You can run it later by calling the `inference()` function.")



# ========================== CLI ==========================
# CUDA_VISIBLE_DEVICES=0 python maltese_xtts.py \
# --output_path checkpoints/ \
# --metadatas datasets-1/metadata_train.csv,datasets-1/metadata_eval.csv,mt datasets-2/metadata_train.csv,datasets-2/metadata_eval.csv,mt \
# --num_epochs 100 \
# --batch_size 8 \
# --grad_acumm 84 \
# --max_text_length 200 \
# --max_audio_length 255995 \
# --weight_decay 1e-2 \
# --lr 5e-6 \
# --save_step 10000

# Based on https://github.com/daswer123/xtts-webui/blob/main/scripts/utils/gpt_train.py & https://github.com/anhnh2002/XTTSv2-Finetuning-for-New-Languages/blob/main/train_gpt_xtts.py