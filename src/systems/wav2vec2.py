import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM
from transformers import HubertForCTC, Data2VecAudioForCTC
from copy import deepcopy
import json

from .base import System
from src.utils.tool import batchify
from .loss import softmax_entropy, mcc_loss, div_loss


class Wav2vec2System(System):

    SAMPLE_RATE = 16000

    def __init__(self, config) -> None:
        self.config = config
        self.history = {}
        self.adapt_count = 0

        # load model and tokenizer
        if config["model_name"] == "patrickvonplaten/wav2vec2-base-960h-4-gram":
            self.processor = Wav2Vec2ProcessorWithLM.from_pretrained(config["model_name"])
        else:
            self.processor = Wav2Vec2Processor.from_pretrained(config["model_name"], sampling_rate=SUTASystem.SAMPLE_RATE)
        
        # Model ablation
        if config["model_name"] == "facebook/data2vec-audio-base-960h":
            self.model = Data2VecAudioForCTC.from_pretrained(config["model_name"])
        elif config["model_name"] == "facebook/hubert-large-ls960-ft":
            self.model = HubertForCTC.from_pretrained(config["model_name"])
        else:
            self.model = Wav2Vec2ForCTC.from_pretrained(config["model_name"])
        
        self.model.train()  # huggingface default loads with eval mode
        self.model.cuda()

        # set up for tent
        self.optimizer, self.scheduler = setup_optimizer(
            self.build_optimized_model(),
            config["opt"], config["lr"], scheduler=config["scheduler"]
        )

        f = open('vocab.json')
        self.vocab = json.load(f)

    def build_optimized_model(self):
        self.model.requires_grad_(False)
        params, self.opt_param_names = self.collect_params()
        # print(param_names[:10])
        for p in params:
            p.requires_grad = True
        print("Optimizable: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        return params

    def _wav_to_model_input(self, wavs):
        # Due to wav2vec2-base special design, no attention mask is returned.
        # Wav2Vec2Processor's default argument for return_attention_mask will be False here.
        # However, it will be True in other speech models!
        inputs = self.processor(
            audio=wavs,
            sampling_rate=Wav2vec2System.SAMPLE_RATE,
            return_tensors="pt",
            padding="longest",
        )
        return inputs.to(device=self.model.device)
    
    def _text_to_model_input(self, texts):
        # target texts need to match wav2vec2's format to make sure correct tokenization
        texts_new = []
        for x in texts:
            x = x.upper()
            x_new = ""
            for s in x:
                if s in self.vocab or s == ' ':
                    x_new += s
            texts_new.append(x_new)

        labels = self.processor(
            text=texts_new,
            return_tensors="pt",
            padding="longest",
        )
        labels = labels.input_ids.masked_fill(labels.attention_mask.ne(1), -100)
        return labels.to(device=self.model.device)

    def l2_loss(self):
        l2_loss = 0.0
        orig_state_dict = self.history["init"][0]

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                l2_loss += torch.sum((param - orig_state_dict[name]) ** 2)
        return l2_loss

    def ctc_adapt(self, wavs, texts, record={}):
        self.adapt_count += 1

        inputs = self._wav_to_model_input(wavs)
        labels = self._text_to_model_input(texts)
        if labels.shape[1] == 0:  # empty string exception, e.g. PL collapse
            labels = torch.zeros((len(labels), 1), device=labels.device)
            record["collapse"] = True
        inputs["labels"] = labels

        outputs = self.model(**inputs)
        loss = outputs.loss
        record["ctc_loss"] = loss.item()
        record["total_loss"] = loss.item()

        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None: 
            self.scheduler.step()
        self.model.zero_grad()

    def ctc_adapt_loss_only(self, wavs, texts, record={}):
        """
        ctc_adapt without gradient control so that we can use gradient accumulation
        """
        self.adapt_count += 1

        inputs = self._wav_to_model_input(wavs)
        labels = self._text_to_model_input(texts)
        if labels.shape[1] == 0:  # empty string exception, e.g. PL collapse
            labels = torch.zeros((len(labels), 1), device=labels.device)
            record["collapse"] = True
        inputs["labels"] = labels

        outputs = self.model(**inputs)
        loss = outputs.loss
        record["ctc_loss"] = loss.item()
        record["total_loss"] = loss.item()

        return loss

    def ctc_adapt_auto(self, wavs, texts, batch_size=-1, record={}) -> None:
        """ ctc_adapt auto split to smaller batch """
        self.adapt_count += 1
        if batch_size == -1:
            batch_size == len(wavs)
        self.model.zero_grad()
        denom_scale = len(wavs) // batch_size
        assert denom_scale > 0
        for wavs, texts in zip(batchify(wavs, batch_size=batch_size), batchify(texts, batch_size=batch_size)):
            loss = self.ctc_adapt_loss_only(wavs, texts, record=record)
            self.adapt_count -= 1  # avoid repeat count
            loss = loss / denom_scale
            loss.backward()
    
        self.optimizer.step()
        self.model.zero_grad()

    @torch.no_grad()
    def inference(self, wavs):
        inputs = self._wav_to_model_input(wavs)
        outputs = self.model(**inputs).logits
        predicted_ids = torch.argmax(outputs, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)
        
        return list(transcription)
    
    @torch.no_grad()
    def beam_inference(self, wavs, n_best=1, text_only=True):
        """ Note that the underlying model should support beam search! """
        inputs = self._wav_to_model_input(wavs)
        logits = self.model(**inputs).logits
        # CAUTION:
        # See https://www.youtube.com/watch?v=mp7fHMTnK9A for definition of alpha and beta, and note that the defualt 
        # value of beta is not 0, which includes word length penalty and therefore not pure LM score
        res = self.processor.batch_decode(logits.cpu().numpy(), n_best=n_best, alpha=0.5, beta=0.0)
        if not text_only:
            return res
        transcription = res.text
        
        return list(transcription)
    
    # def save(self, path: str) -> None:
    #     torch.save(self.model.state_dict(), path)

    # def load(self, path: str) -> None:
    #     self.model.load_state_dict(torch.load(path))
    
    def collect_params(self):
        """Collect the affine scale + shift parameters from batch norms.

        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.

        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        trainable = []
        if self.config["bias_only"]:
            trainable = ['bias']
        else: 
            trainable = ['weight', 'bias']

        if self.config.get("bitfit", False):
            for np, p in self.model.named_parameters():
                if str(np).split('.')[1] == 'encoder' and "bias" in np:
                    p.requires_grad = True
                    params.append(p)
                    names.append(np)
        
        for nm, m in self.model.named_modules():
            # print(nm)
            if self.config["train_LN"]: 
                if isinstance(m, nn.LayerNorm):
                    for np, p in m.named_parameters():
                        if np in trainable:
                            if not p.requires_grad:
                                p.requires_grad = True
                                params.append(p)
                                names.append(f"{nm}.{np}")
            if self.config["train_feature"]:
                if len(str(nm).split('.')) > 1:
                    if str(nm).split('.')[1] == 'feature_extractor' or str(nm).split('.')[1] == 'feature_projection':
                        for np, p in m.named_parameters():
                            p.requires_grad = True
                            params.append(p)
                            names.append(f"{nm}.{np}")
                            
            if self.config["train_all"]: 
                for np, p in m.named_parameters():
                    p.requires_grad = True
                    params.append(p)
                    names.append(f"{nm}.{np}")

        return params, names


def setup_optimizer(params, opt_name='AdamW', lr=1e-4, beta=0.9, weight_decay=0., scheduler=None, step_size=1, gamma=0.7):
    opt = getattr(torch.optim, opt_name)
    print(f'[INFO]    optimizer: {opt}')
    print(f'[INFO]    scheduler: {scheduler}')
    if opt_name == 'Adam':       
        optimizer = opt(params,
                lr=lr,
                betas=(beta, 0.999),
                weight_decay=weight_decay)
    else: 
        optimizer = opt(params, lr=lr, weight_decay=weight_decay)
    
    if scheduler is not None: 
        return optimizer, eval(scheduler)(optimizer, step_size=step_size, gamma=gamma)
    else: 
        return optimizer, None
