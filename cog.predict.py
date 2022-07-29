from asyncio import subprocess
import os
import multiprocessing
#os.environ["DISCOART_LOG_LEVEL"] = 'DEBUG'
os.environ["DISCOART_OPTOUT_CLOUD_BACKUP"] = '1'
os.environ["DISCOART_DISABLE_IPYTHON"] = '1'
os.environ["DISCOART_DISABLE_RESULT_SUMMARY"] = '1'
os.environ["DISCOART_DISABLE_REMOTE_MODELS"] = '1'
os.environ["DISCOART_CACHE_DIR"] = '/root/.cache/discoart'

import tempfile
from cog import BasePredictor, Input, Path
from types import SimpleNamespace
from typing import Iterator
import subprocess


import threading
from queue import SimpleQueue
from docarray import DocumentArray, Document

import yaml
from yaml import Loader

with open(
    f'/src/discoart/resources/models.yml'
) as ymlfile:
    model_list = yaml.load(ymlfile, Loader=Loader)
del model_list["secondary"]

default_clip_models = [
    'ViT-B-32::openai',
    'ViT-B-16::openai',
    'RN50::openai'
]


class Predictor(BasePredictor):
    def setup(self):
        subprocess.call("python setup.py develop", shell=True)
        
        from discoart.helper import (
            load_diffusion_model,
            load_clip_models,
            load_secondary_model,
            get_device,
            free_memory,            
        )
        from discoart.config import (
            load_config,
            default_args
        )

        self.clip_models_cache = {}
        self.device = get_device()
        self.args = SimpleNamespace(**load_config(default_args))
        self.model, self.diffusion = load_diffusion_model(
            self.args, self.device)
        self.clip_model_list = default_clip_models
        self.clip_models = load_clip_models(
            self.device, self.clip_model_list, self.clip_models_cache, text_clip_on_cpu=False)
        self.secondary_model = load_secondary_model(
            self.args, device=self.device)
        free_memory()

    def predict(
        self,
        steps: int = Input(
            description="Number of steps, higher numbers will give more refined output but will take longer", default=100),
        prompt: str = Input(description="Text Prompt",
                            default="A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation."),
        width: int = Input(
            description="Width of the output image, higher numbers will take longer", default=1280),
        height: int = Input(
            description="Height of the output image, higher numbers will take longer", default=768),
        diffusion_model: str = Input(description="Diffusion Model", default="512x512_diffusion_uncond_finetune_008100", choices=model_list.keys()),
        diffusion_sampling_mode: str = Input(
            description="Diffusion Sampling Mode", default="ddim", choices=["plms", "ddim"]),
        ViTB32: bool = Input(description="Use ViTB32 model", default=True),
        ViTB16: bool = Input(description="Use ViTB16 model", default=True),
        ViTL14: bool = Input(description="Use ViTB14 model", default=False),
        ViTL14_336: bool = Input(
            description="Use ViTL14_336 model", default=False),
        RN50: bool = Input(description="Use RN50 model", default=True),
        RN50x4: bool = Input(description="Use RN50x4 model", default=False),
        RN50x16: bool = Input(description="Use RN50x16 model", default=False),
        RN50x64: bool = Input(description="Use RN50x64 model", default=False),
        RN50x101: bool = Input(
            description="Use RN50x101 model", default=False),
        RN101: bool = Input(description="Use RN101 model", default=False),
        ViTB32_laion2b_e16: bool = Input(
            description="Use ViTB32_laion2b_e16 model", default=False),
        ViTB32_laion400m_e31: bool = Input(
            description="Use ViTB32_laion400m_e31 model", default=False),
        ViTB32_laion400m_e32: bool = Input(
            description="Use ViTB32_laion400m_e32 model", default=False),
        ViTB32quickgelu_laion400m_e31: bool = Input(
            description="Use ViTB32quickgelu_laion400m_e31 model", default=False),
        ViTB32quickgelu_laion400m_e32: bool = Input(
            description="Use ViTB32quickgelu_laion400m_e32 model", default=False),
        ViTB16_laion400m_e31: bool = Input(
            description="Use ViTB16_laion400m_e31 model", default=False),
        ViTB16_laion400m_e32: bool = Input(
            description="Use ViTB16_laion400m_e32 model", default=False),
        RN50_yffcc15m: bool = Input(
            description="Use RN50_yffcc15m model", default=False),
        RN50_cc12m: bool = Input(
            description="Use RN50_cc12m model", default=False),
        RN50_quickgelu_yfcc15m: bool = Input(
            description="Use RN50_quickgelu_yfcc15m model", default=False),
        RN50_quickgelu_cc12m: bool = Input(
            description="Use RN50_quickgelu_cc12m model", default=False),
        RN101_yfcc15m: bool = Input(
            description="Use RN101_yfcc15m model", default=False),
        RN101_quickgelu_yfcc15m: bool = Input(
            description="Use RN101_quickgelu_yfcc15m model", default=False),
        use_secondary_model: bool = Input(
            description="Use secondary model", default=True),
        clip_guidance_scale: int = Input(
            description="CLIP Guidance Scale", default=5000),
        tv_scale: int = Input(description="TV Scale", default=0),
        range_scale: int = Input(description="Range Scale", default=150),
        sat_scale: int = Input(description="Saturation Scale", default=0),
        cutn_batches: int = Input(description="Cut Batches", default=4),
        skip_augs: bool = Input(
            description="Skip Augmentations", default=False),
        init_image: Path = Input(
            description="Initial image to start generation from", default=None),
        target_image: Path = Input(
            description="Target image to generate towards, similarly to the text prompt", default=None),
        init_scale: int = Input(description="Initial Scale", default=1000),
        target_scale: int = Input(description="Target Scale", default=20000),
        skip_steps: int = Input(description="Skip Steps", default=10),
        display_rate: int = Input(
            description="Steps between outputs, lower numbers may slow down generation.", default=20),
        seed: int = Input(
            description="Seed (leave empty to use a random seed)", default=None, le=(2**32-1), ge=0),
    ) -> Iterator[Path]:      
      from discoart.helper import (
          load_diffusion_model,
          load_clip_models,
          free_memory,          
          show_result_summary,
          get_output_dir,
      )
      from discoart.config import ( load_config, default_args )

      self.args = SimpleNamespace(**load_config(default_args))
      self.args.batch_size = 1
      self.args.n_batches = 1
      if seed:
        self.args.seed = seed
      self.args.diffusion_sampling_mode = diffusion_sampling_mode
      self.args.use_secondary_model = use_secondary_model
      self.args.clip_guidance_scale = clip_guidance_scale
      self.args.tv_scale = tv_scale
      self.args.range_scale = range_scale
      self.args.sat_scale = sat_scale
      self.args.cutn_batches = cutn_batches
      self.args.skip_augs = skip_augs
      self.args.display_rate = display_rate
      if init_image:
        self.args.init_image = str(init_image)
      else:
        self.args.init_image = None
      if target_image:
        self.args.target_image = str(target_image)
      else:
        self.args.target_image = None
      self.args.init_scale = init_scale
      self.args.target_scale = target_scale
      self.args.skip_steps = skip_steps
      self.args.steps = steps
      self.args.width_height = [width, height]
      self.args.text_prompts = prompt

      if (diffusion_model != self.args.diffusion_model):
        self.args.diffusion_model = diffusion_model
        self.model, self.diffusion = load_diffusion_model(
            self.args, self.device)

      clip_models = []
      if ViTB32:
        clip_models.append("ViT-B-32::openai")
      if ViTB16:
        clip_models.append("ViT-B-16::openai")
      if ViTL14:
        clip_models.append("ViT-L-14::openai")
      if ViTL14_336:
        clip_models.append("ViT-L-14::openai")       
      if RN50:
        clip_models.append("RN50::openai")
      if RN50x4:
        clip_models.append("RN50x4::openai")
      if RN50x16:
        clip_models.append("RN50x16::openai")
      if RN50x64:
        clip_models.append("RN50x64::openai")
      if RN50x101:
        clip_models.append("RN50x101::openai")
      if RN101:
        clip_models.append("RN101::openai")
      if ViTB32_laion2b_e16:
        clip_models.append("ViT-B-32::laion2b_e16")
      if ViTB32_laion400m_e31:
        clip_models.append("ViT-B-32::laion400m_e31")
      if ViTB32_laion400m_e32:
        clip_models.append("ViT-B-32::laion400m_e32")
      if ViTB32quickgelu_laion400m_e31:
        clip_models.append("ViT-B-32-quickgelu::laion400m_e31")
      if ViTB32quickgelu_laion400m_e32:
        clip_models.append("ViT-B-32-quickgelu::laion400m_e32")
      if ViTB16_laion400m_e31:
        clip_models.append("ViT-B-16::laion400m_e31")
      if ViTB16_laion400m_e32:
        clip_models.append("ViT-B-16::laion400m_e32")
      if RN50_yffcc15m:
        clip_models.append("RN50::yffcc15m")
      if RN50_cc12m:
        clip_models.append("RN50::cc12m")
      if RN50_quickgelu_yfcc15m:
        clip_models.append("RN50-quickgelu::yfcc15m")
      if RN50_quickgelu_cc12m:
        clip_models.append("RN50-quickgelu::cc12m")
      if RN101_yfcc15m:
        clip_models.append("RN101::yfcc15m")
      if RN101_quickgelu_yfcc15m:
        clip_models.append("RN101-quickgelu::yfcc15m")
      
      if (clip_models != self.clip_model_list):
        self.clip_model_list = clip_models
        self.clip_models = load_clip_models(
            self.device, self.clip_model_list, self.clip_models_cache, text_clip_on_cpu=False)

      try:
        self.stop_event = multiprocessing.Event()
        self.skip_event = multiprocessing.Event()
        output = SimpleQueue()
        self.image_callback = lambda img: output.put(img)
        t = threading.Thread(target=self.worker, daemon=True)
        t.start()
        while t.is_alive():
          try:
            image = output.get(block=True, timeout=5)
            yield Path(image)
          except:
              {}
      except:
        self.stop_event.set()
      finally:
        free_memory()
        _name = self.args.name_docarray
        pb_path = os.path.join(get_output_dir(_name), 'da.protobuf.lz4')
        if os.path.exists(pb_path):
          _da = DocumentArray.load_binary(pb_path)
          tmp = tempfile.mkdtemp()
          result_file = f'{tmp}/discoart-result.png'
          _da[0].save_uri_to_file(result_file)
          yield(Path(result_file))

    def worker(self):
      from discoart.runner import do_run
      da = do_run(
          self.args,
          (self.model, self.diffusion, self.clip_models, self.secondary_model),
          device=self.device,
          events=(self.skip_event, self.stop_event),
          image_callback=self.image_callback
      )
