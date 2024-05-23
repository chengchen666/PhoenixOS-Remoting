# python trainer.py --model_path=/tmp/model --config config/test.yaml
import sys
import ctypes
import os
import time

# Hide welcome message from bitsandbytes
os.environ.update({
    "BITSANDBYTES_NOWELCOME": "1",
    "DIFFUSERS_VERBOSITY": "error"
})

import torch
import lightning.pytorch as pl

from lib.callbacks import HuggingFaceHubCallback, SampleCallback
from lib.model import StableDiffusionModel, get_pipeline
from lib.compat import pl_compat_fix
from lib.precision import HalfPrecisionPlugin

from omegaconf import OmegaConf
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import SingleDeviceStrategy
from lightning.pytorch import seed_everything
from lib.utils import rank_zero_print

def main():
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    
    # load remoting bottom library
    path = os.getenv('REMOTING_BOTTOM_LIBRARY')
    if path is not None:
        cpp_lib = ctypes.CDLL(path)
        start_trace = cpp_lib.startTrace
        end_trace = cpp_lib.endTrace

    if(len(sys.argv) < 3):
        print('Usage: python3 train.py num_iter batch_size')
        sys.exit()

    num_iter = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    
    config = OmegaConf.load(script_dir + "/config.yaml")
    config.trainer.model_path = script_dir + "/" + config.trainer.model_path
    config.trainer.batch_size = batch_size
    config.dataset.img_path = script_dir + "/" + config.dataset.img_path
    seed_everything(config.trainer.seed)

    model_path = config.trainer.model_path

    strategy = None
    if config.lightning.accelerator in ["gpu", "cpu"]:
        strategy = "ddp"

    if config.trainer.use_hivemind:
        from lib.hivemind import init_hivemind
        strategy = init_hivemind(config)

    rank_zero_print(f"Loading model from {model_path}")
    pipeline = get_pipeline(model_path)
    if config.get("lora"):
        if config.lora.get("use_locon"):
            from experiment.locon import LoConDiffusionModel
            model = LoConDiffusionModel(pipeline, config)
        else:
            from experiment.lora import LoRADiffusionModel
            model = LoRADiffusionModel(pipeline, config)
        strategy = config.lightning.strategy = "auto"
    else:
        model = StableDiffusionModel(pipeline, config)

    major, minor = torch.__version__.split('.')[:2]
    if (int(major) > 1 or (int(major) == 1 and int(minor) >= 12)) and torch.cuda.is_available():
        device = torch.cuda.get_device_properties(0)
        compute_capability = float(f"{device.major}.{device.minor}")
        precision = 'high' if config.lightning.precision == 32 else 'medium'
        if compute_capability >= 8.0:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision(precision)

    callbacks = []
    if config.monitor.huggingface_repo != "":
        hf_logger = HuggingFaceHubCallback(
            repo_name=config.monitor.huggingface_repo,
            use_auth_token=config.monitor.hf_auth_token,
            **config.monitor
        )
        callbacks.append(hf_logger)

    logger = None
    if config.monitor.wandb_id != "":
        logger = WandbLogger(project=config.monitor.wandb_id)
        callbacks.append(LearningRateMonitor(logging_interval='step'))

    if config.get("custom_embeddings") != None and config.custom_embeddings.enabled:
        from experiment.textual_inversion import CustomEmbeddingsCallback
        callbacks.append(CustomEmbeddingsCallback(config.custom_embeddings))

    if config.get("sampling") != None and config.sampling.enabled:
        callbacks.append(SampleCallback(config.sampling, logger))

    if torch.cuda.device_count() == 1:
        strategy = SingleDeviceStrategy(device="cuda:0")

    if config.lightning.get("strategy") is not None:
        strategy = config.lightning.strategy
        del config.lightning["strategy"]

    if not config.get("custom_embeddings") or not config.custom_embeddings.freeze_unet:
        checkpoint_config = {
            k: v
            for k, v in config.checkpoint.items() if k != "extended"
        }
        callbacks.append(ModelCheckpoint(**checkpoint_config))
        enable_checkpointing = True
    else:
        enable_checkpointing = False

    if config.lightning.get("enable_checkpointing") == None:
        config.lightning.enable_checkpointing = enable_checkpointing
        
    plugins = None
    target_precision = config.lightning.precision
    if target_precision in ["16-true", "bf16-true"]:
        plugins = HalfPrecisionPlugin(target_precision)
        model.to(torch.float16 if target_precision == "16-true" else torch.bfloat16)
        del config.lightning.precision

    if path is not None:
        start_trace()
    
    T1 = time.time()

    config.lightning.max_epochs = num_iter

    # config.lightning.replace_sampler_ddp = False
    config, callbacks = pl_compat_fix(config, callbacks)
    trainer = pl.Trainer(
        logger=logger, 
        callbacks=callbacks, 
        strategy=strategy, 
        plugins=plugins, 
        **config.lightning
    )
    trainer.fit(model=model, ckpt_path=None)
    
    T2 = time.time()
    print('time used: ', T2-T1)

    if path is not None:
        end_trace()

if __name__ == "__main__":
    main()

