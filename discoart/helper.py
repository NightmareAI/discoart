import gc
import hashlib
import json
import logging
import os
import sys
import threading
import urllib.parse
import urllib.request
import warnings
from os.path import expanduser
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any, List, Tuple
from urllib.request import Request, urlopen

import pkg_resources
import regex as re
import torch
import yaml
from clip.simple_tokenizer import SimpleTokenizer, whitespace_clean, basic_clean
from packaging.version import Version

from spellchecker import SpellChecker
from tqdm.auto import tqdm

cache_dir = os.environ.get(
    'DISCOART_CACHE_DIR', os.path.join(expanduser('~'), '.cache', __package__)
)

from yaml import Loader

from . import __resources_path__


def _get_logger():
    logger = logging.getLogger(__package__)
    _log_level = os.environ.get('DISCOART_LOG_LEVEL', 'INFO')
    logger.setLevel(_log_level)
    ch = logging.StreamHandler()
    ch.setLevel(_log_level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    ch.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(ch)
    logger.propagate = False
    return logger


logger = _get_logger()


def get_model_list():
    with open(
        os.environ.get('DISCOART_MODELS_YAML', f'{__resources_path__}/models.yml')
    ) as ymlfile:
        return yaml.load(ymlfile, Loader=Loader)


models_list = get_model_list()


def get_remote_model_list(local_model_list: Dict[str, Any], force_print: bool = False):
    if 'DISCOART_DISABLE_REMOTE_MODELS' in os.environ:
        return

    try:
        req = Request(
            os.environ.get(
                'DISCOART_REMOTE_MODELS_URL',
                'https://raw.githubusercontent.com/jina-ai/discoart/main/discoart/resources/models.yml',
            ),
            headers={'User-Agent': 'Mozilla/5.0'},
        )
        with urlopen(
            req, timeout=2
        ) as resp:  # 'with' is important to close the resource after use
            remote_model_list = yaml.load(resp, Loader=Loader)
    except Exception as ex:
        logger.error(f'can not fetch the latest `model_list` from remote, {ex}')

    if (remote_model_list and remote_model_list != local_model_list) or force_print:
        if not force_print:
            logger.warning(
                'remote model list is different from the local model list, override local list.'
            )

        from rich.table import Table
        from rich import box, print

        param_str = Table(
            box=box.ROUNDED,
            highlight=True,
            title_justify='center',
        )
        param_str.add_column('Diffusion Model', justify='right')
        param_str.add_column('Remote', justify='left')
        param_str.add_column('Local', justify='left')
        param_str.add_column('Synced', justify='left')
        all_models = sorted(
            set(list(remote_model_list.keys()) + list(local_model_list.keys()))
        )
        for k in all_models:
            param_str.add_row(
                k,
                str(k in remote_model_list),
                str(k in local_model_list),
                str(remote_model_list.get(k) == local_model_list.get(k)),
            )

        print(param_str)
        local_model_list.clear()
        local_model_list.update(remote_model_list)


threading.Thread(
    target=get_remote_model_list, args=(models_list, False), daemon=True
).start()


def get_device():
    # check if GPU is available

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
        warnings.warn(
            '''
!!!!CUDA is not available. DiscoArt is running on CPU. `create()` will be unbearably slow on CPU!!!!
Please switch to a GPU device. If you are using Google Colab, then free tier would just work.
            '''
        )
    return device


def is_jupyter() -> bool:  # pragma: no cover
    """
    Check if we're running in a Jupyter notebook, using magic command `get_ipython` that only available in Jupyter.

    :return: True if run in a Jupyter notebook else False.
    """
    if 'DISCOART_DISABLE_IPYTHON' in os.environ:
        return False

    try:
        get_ipython  # noqa: F821
    except NameError:
        return False
    shell = get_ipython().__class__.__name__  # noqa: F821
    if shell == 'ZMQInteractiveShell':
        return True  # Jupyter notebook or qtconsole
    elif shell == 'Shell':
        return True  # Google colab
    elif shell == 'TerminalInteractiveShell':
        return False  # Terminal running IPython
    else:
        return False  # Other type (?)


def is_google_colab() -> bool:  # pragma: no cover
    if 'DISCOART_DISABLE_IPYTHON' in os.environ:
        return False

    try:
        get_ipython  # noqa: F821
    except NameError:
        return False
    shell = get_ipython().__class__.__name__  # noqa: F821
    return shell == 'Shell'


def get_ipython_funcs(show_widgets: bool = False):
    class NOP:
        def __call__(self, *args, **kwargs):
            return NOP()

        __getattr__ = __enter__ = __exit__ = __call__

    if is_jupyter():
        from IPython import display as dp1
        from IPython.display import FileLink as fl

        handlers = None

        if show_widgets:
            from ipywidgets import HTML, IntProgress, Textarea, Tab

            pg_bar = IntProgress(
                value=1,
                min=0,
                max=4,
                step=1,
                description=f'n_batches:',
                bar_style='info',  # 'success', 'info', 'warning', 'danger' or ''
                orientation='horizontal',
            )
            html_handle = HTML()

            nondefault_config_handle = HTML()
            all_config_handle = HTML()
            code_snippet_handle = Textarea(rows=20)
            tab = Tab()
            tab.children = [
                html_handle,
                nondefault_config_handle,
                all_config_handle,
                code_snippet_handle,
            ]
            for idx, j in enumerate(
                ('Preview', 'Non-default config', 'Full config', 'Code snippet')
            ):
                tab.set_title(idx, j)

            handlers = SimpleNamespace(
                preview=html_handle,
                config=nondefault_config_handle,
                all_config=all_config_handle,
                code=code_snippet_handle,
                progress=pg_bar,
            )

        def redraw():
            dp1.display(pg_bar, tab)

        return dp1, fl, handlers, redraw
    else:
        return NOP(), NOP(), NOP(), NOP()


if not os.path.exists(cache_dir):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

logger.debug(f'`.cache` dir is set to: {cache_dir}')

check_model_SHA = False


def _wget(url, outputdir):
    logger.debug(f'downloading from {url}...')
    try:
        basename = os.path.basename(url)

        with urllib.request.urlopen(url) as source, open(
            os.path.join(outputdir, basename), 'wb'
        ) as output:
            with tqdm(
                total=int(source.info().get("Content-Length")),
                ncols=80,
                unit='iB',
                unit_scale=True,
            ) as loop:
                while True:
                    buffer = source.read(8192)
                    if not buffer:
                        break

                    output.write(buffer)
                    loop.update(len(buffer))
        logger.debug(f'write to {outputdir}')
    except:
        logger.error(f'failed to download {url}')


def load_clip_models(
    device,
    enabled: List[str],
    clip_models: Dict[str, Any] = {},
    text_clip_on_cpu: bool = False,
):
    logger.debug('loading clip models...')

    if text_clip_on_cpu:
        first_device = torch.device('cpu')
        logger.debug(f'CLIP will be first loaded to CPU')
    else:
        first_device = device

    # load enabled models
    for k in enabled:
        if k not in clip_models:
            if '::' in k and k.split('::')[-1] != 'openai':
                # use open_clip loader
                k1, k2 = k.split('::')
                logger.debug(f'use open_clip to load {k1}')
                import open_clip

                m = open_clip.create_model_and_transforms(
                    k1,
                    device=first_device,
                    pretrained=k2,
                )[0]
            else:

                k1 = (
                    k.replace('B-32::', 'B/32::')
                    .replace('B-16::', 'B/16::')
                    .replace('L-14::', 'L/14::')
                    .replace('L-14-336::', 'L/14@336px::')
                )

                if '::' in k1:
                    k1 = k1.split('::')[0]

                logger.debug(f'use openai clip to load {k1}')

                import clip

                m = clip.load(k1, device=first_device, jit=False)[0]
            clip_models[k] = m.eval().requires_grad_(False)
            if text_clip_on_cpu:
                # then the first device is CPU, we now load visual arm back to GPU.
                m.visual.to(device)
                logger.debug(f'move {k}.visual to GPU')

    # disable not enabled models to save memory
    for k in list(clip_models.keys()):
        if k not in enabled:
            clip_models.pop(k)

    return clip_models


def _check_sha(path, expected_sha):
    if 'DISCOART_CHECK_MODEL_SHA' in os.environ:
        with open(path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest() == expected_sha
    else:
        return True


def _get_model_name(name: str) -> str:
    for k in models_list.keys():
        if k.lower().startswith(name.lower().strip()):
            return k


def download_model(model_name: str):
    if os.path.isfile(model_name):
        logger.debug('use customized local model')
        return

    model_name = _get_model_name(model_name)

    if not model_name:
        raise ValueError(
            f'{model_name} is not supported, must be one of {models_list.keys()}'
        )

    model_filename = os.path.basename(models_list[model_name]['sources'][0])
    model_local_path = os.path.join(cache_dir, model_filename)
    if os.path.exists(model_local_path) and _check_sha(
        model_local_path, models_list[model_name]['sha']
    ):
        logger.debug(f'{model_filename} is already downloaded with correct SHA')
    else:
        for url in models_list[model_name]['sources']:
            _wget(url, cache_dir)
            if _check_sha(model_local_path, models_list[model_name]['sha']):
                logger.debug(f'{model_filename} is downloaded with correct SHA')
                break


def get_diffusion_config(user_args, device=torch.device('cuda:0')) -> Dict[str, Any]:
    diffusion_model = user_args.diffusion_model
    steps = user_args.steps
    diffusion_config = user_args.diffusion_model_config

    from guided_diffusion.script_util import (
        model_and_diffusion_defaults,
    )

    model_config = model_and_diffusion_defaults()

    _diff_model_name = _get_model_name(diffusion_model)
    if _diff_model_name and models_list[_diff_model_name].get('config', None):
        model_config.update(models_list[_diff_model_name]['config'])
    else:
        logger.info(
            '''
        looks like you are using a custom diffusion model, 
        to override default diffusion model config, you can specify `create(diffusion_model_config={...}, ...)` as well,
        '''
        )
        model_config.update(
            {
                'attention_resolutions': '16',
                'class_cond': False,
                'diffusion_steps': 1000,
                'rescale_timesteps': True,
                'timestep_respacing': 'ddim100',
                'image_size': 256,
                'learn_sigma': True,
                'noise_schedule': 'linear',
                'num_channels': 128,
                'num_heads': 1,
                'num_res_blocks': 2,
                'use_scale_shift_norm': False,
            }
        )
    timestep_respacing = f'ddim{steps}'
    diffusion_steps = (1000 // steps) * steps if steps < 1000 else steps

    model_config.update(
        {
            'use_fp16': device.type != 'cpu',
            'timestep_respacing': timestep_respacing,
            'diffusion_steps': diffusion_steps,
            'use_checkpoint': True,
        }
    )

    if diffusion_config and isinstance(diffusion_config, dict):
        model_config.update(diffusion_config)

    return model_config


def load_secondary_model(user_args, device=torch.device('cuda:0')):
    if not user_args.use_secondary_model:
        return
    download_model('secondary')

    from discoart.nn.sec_diff import SecondaryDiffusionImageNet2

    secondary_model = SecondaryDiffusionImageNet2()
    secondary_model.load_state_dict(
        torch.load(
            os.path.join(cache_dir, 'secondary_model_imagenet_2.pth'),
            map_location='cpu',
        )
    )
    secondary_model.eval().requires_grad_(False).to(device)
    return secondary_model


def load_diffusion_model(user_args, device):
    diffusion_model = user_args.diffusion_model

    _diff_model_name = _get_model_name(diffusion_model)
    if _diff_model_name:
        rec_size = models_list[_diff_model_name].get('recommended_size', None)
        if rec_size and user_args.width_height != rec_size:
            logger.warning(
                f'{diffusion_model} is recommended to have width_height {rec_size}, but you are using {user_args.width_height}. This may lead to suboptimal results.'
            )

    download_model(diffusion_model)

    model_config = get_diffusion_config(user_args, device=device)

    logger.debug('loading diffusion model...')
    from guided_diffusion.script_util import (
        create_model_and_diffusion,
    )

    model, diffusion = create_model_and_diffusion(**model_config)
    if os.path.isfile(diffusion_model):
        logger.debug(f'loading customized diffusion model from {diffusion_model}')
        _model_path = diffusion_model
    elif _diff_model_name:
        model_filename = os.path.basename(models_list[_diff_model_name]['sources'][0])
        _model_path = os.path.join(cache_dir, model_filename)
    model.load_state_dict(torch.load(_model_path, map_location='cpu'), strict=False)
    model.requires_grad_(False).eval().to(device)

    for name, param in model.named_parameters():
        if 'qkv' in name or 'norm' in name or 'proj' in name:
            param.requires_grad_()
    if model_config['use_fp16']:
        model.convert_to_fp16()

    return model, diffusion


class PromptParser(SimpleTokenizer):
    def __init__(self, on_misspelled_token: str, **kwargs):
        super().__init__(**kwargs)
        self.spell = SpellChecker()
        from . import __resources_path__

        with open(
            os.path.join(__resources_path__, 'vocab.txt'), encoding='utf-8'
        ) as fp:
            self.spell.word_frequency.load_words(
                line.strip() for line in fp if len(line.strip()) > 1
            )
        self.on_misspelled_token = on_misspelled_token

    @staticmethod
    def _split_weight(prompt):
        if ':' in prompt:
            vals = prompt.rsplit(':', 1)
        else:
            vals = [prompt, 1]
        return vals[0], float(vals[1])

    def parse(self, text: str, on_misspelled_token=None) -> Tuple[str, float]:
        text, weight = self._split_weight(text)
        text = whitespace_clean(basic_clean(text)).lower()
        all_tokens = []
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            all_tokens.append(token)
        unknowns = [
            v
            for v in self.spell.unknown(all_tokens)
            if len(v) > 2 and self.spell.correction(v) != v
        ]
        if unknowns:
            pairs = []
            for v in unknowns:
                vc = self.spell.correction(v)
                pairs.append((v, vc))
                on_misspelled_token = on_misspelled_token or self.on_misspelled_token
                if on_misspelled_token == 'correct':
                    for idx, ov in enumerate(all_tokens):
                        if ov == v:
                            all_tokens[idx] = vc

            if pairs:
                warning_str = '\n'.join(
                    f'Misspelled `{v}`, do you mean `{vc}`?' for v, vc in pairs
                )
                if on_misspelled_token == 'raise':
                    raise ValueError(warning_str)
                elif on_misspelled_token == 'correct':
                    logger.warning(
                        'auto-corrected the following tokens:\n' + warning_str
                    )
                else:
                    logger.warning(
                        'Found misspelled tokens in the prompt:\n' + warning_str
                    )

        logger.debug(f'prompt: {all_tokens}, weight: {weight}')
        return ' '.join(all_tokens), weight


def free_memory():
    gc.collect()
    torch.cuda.empty_cache()


def show_result_summary(_da, _name, _args):
    from .config import print_args_table

    _dp1, _fl = get_ipython_funcs()[:2]

    _dp1.clear_output(wait=True)

    imcomplete_str = ''

    fully_done = sum(bool(j) for j in _da[:, 'tags___status__completed'])
    num_expected = _args.n_batches * _args.batch_size
    if _da and fully_done < num_expected:
        imcomplete_str = f'''
# ⚠️ Incomplete result ({fully_done}/{num_expected})

Your `n_batches={_args.n_batches}, batch_size={_args.batch_size}` so supposedly {num_expected} images will be generated, 
but only {fully_done} images were fully completed. This may due to the following reasons:
- You cancel the process before it finishes;
- (On Google Colab) your GPU session is expired;

To avoid this, you can set `n_batches` to a smaller number in `create()`, say `create(n_batches=1)`.
'''

    from rich.markdown import Markdown

    md = Markdown(
        f'''
{imcomplete_str}

# 👀 Result preview

This preview is **NOT** in HD. Do **NOT** use it for your final artworks.

To save the full-size images, please check out the instruction in the next section.
    ''',
        code_theme='igor',
    )
    _dp1.display(md)

    if _da and _da[0].uri:
        _da.plot_image_sprites(skip_empty=True, show_index=True, keep_aspect_ratio=True)

    print_args_table(vars(_args))

    persist_file = _fl(
        os.path.join(get_output_dir(_name), 'da.protobuf.lz4'),
        result_html_prefix=f'▶ Download the local backup (in case cloud storage failed): ',
    )

    md = Markdown(
        f'''


# 🖼️ Save images

Final results and intermediate results are created, i.e.
```text
./{_name}/[i]-done.png
./{_name}/[i]-step-[i].png
./{_name}/[i]-progress.gif
./{_name}/[i]-progress.png
./{_name}/da.protobuf.lz4
```

where:

- `i-*` is up to the value of `n_batches`.
- `*-done-*` is the final image on done.
- `*-step-*` is the intermediate image at certain step.
- `*-progress.png` is the sprite image of all intermediate results so far.
- `*-progress.gif` is the animated gif of all intermediate results so far.
- `da.protobuf.lz4` is the LZ4 compressed Protobuf file of all intermediate results of all `n_batches`.

# 💾 Save & load the batch        

Results are stored in a [DocumentArray](https://docarray.jina.ai/fundamentals/documentarray/) available both local and cloud.


You may also download the file manually and load it from local disk:

```python
da = DocumentArray.load_binary('{get_output_dir(_name)}/da.protobuf.lz4')
```

You can simply pull it from any machine:

```python
# pip install docarray[common]
from docarray import DocumentArray

da = DocumentArray.pull('{_name}')
```

More usage such as plotting, post-analysis can be found in the [README](https://github.com/jina-ai/discoart).
            ''',
        code_theme='igor',
    )
    if is_google_colab():
        _dp1.display(md)
    else:
        _dp1.display(persist_file, md)


def list_diffusion_models():
    get_remote_model_list(models_list, force_print=True)


def _version_check(package: str = None, github_repo: str = None):
    try:
        if not package:
            package = vars(sys.modules[__name__])['__package__']
        if not github_repo:
            github_repo = package

        cur_ver = Version(pkg_resources.get_distribution(package).version)
        req = Request(
            f'https://pypi.python.org/pypi/{package}/json',
            headers={'User-Agent': 'Mozilla/5.0'},
        )
        with urlopen(
            req, timeout=1
        ) as resp:  # 'with' is important to close the resource after use
            j = json.load(resp)
            releases = j.get('releases', {})
            latest_release_ver = max(
                Version(v) for v in releases.keys() if '.dev' not in v
            )
            if cur_ver < latest_release_ver:
                from rich import print
                from rich.panel import Panel

                print(
                    Panel(
                        f'You are using [b]{package} {cur_ver}[/b], but [bold green]{latest_release_ver}[/] is available. '
                        f'You may upgrade it via [b]pip install -U {package}[/b]. [link=https://github.com/jina-ai/{github_repo}/releases]Read Changelog here[/link].',
                        title=':new: New version available!',
                        width=50,
                    )
                )
    except Exception as ex:
        # no network, too slow, PyPi is down
        logger.error(f'can not fetch the lastest version number: {ex}')


threading.Thread(target=_version_check, args=(__package__, 'discoart')).start()
_MAX_DIFFUSION_STEPS = 1000


def _eval_scheduling_str(val) -> List[float]:
    if isinstance(val, str):
        r = eval(val)
    elif isinstance(val, (int, float, bool)):
        r = [val] * _MAX_DIFFUSION_STEPS
    else:
        raise ValueError(f'unsupported scheduling type: {val}: {type(val)}')

    if len(r) != _MAX_DIFFUSION_STEPS:
        raise ValueError(
            f'invalid scheduling string: {val} the schedule steps should be exactly {_MAX_DIFFUSION_STEPS}'
        )
    return r


def _get_current_schedule(schedule_table: Dict, t: int) -> 'SimpleNamespace':
    return SimpleNamespace(**{k: schedule_table[k][t] for k in schedule_table.keys()})


def _get_schedule_table(args) -> Dict:
    return {
        k: _eval_scheduling_str(getattr(args, k))
        for k in (
            'cut_overview',
            'cut_innercut',
            'cut_icgray_p',
            'cut_ic_pow',
            'use_secondary_model',
            'cutn_batches',
            'skip_augs',
            'clip_guidance_scale',
            'tv_scale',
            'range_scale',
            'sat_scale',
            'init_scale',
            'clamp_grad',
            'clamp_max',
        )
    }


def get_output_dir(name_da):
    output_dir = os.path.join(os.environ.get('DISCOART_OUTPUT_DIR', './'), name_da)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir
