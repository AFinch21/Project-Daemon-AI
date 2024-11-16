from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import  TextStreamer

class ModelInference:
    
    def __init__(self, model_id, model_kwargs=None, tokenizer_kwargs=None, streamer_kwargs=None, revision="main"):
        self.model_id = model_id
        self.revision = revision
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.tokenizer_kwargs = tokenizer_kwargs if tokenizer_kwargs is not None else {}
        self.streamer_kwargs = streamer_kwargs if streamer_kwargs is not None else {}
        
        # Load model and tokenizer
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
        self.streamer = self.load_streamer()

    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            trust_remote_code=False,
            revision=self.revision,
            **self.model_kwargs  # Pass additional keyword arguments
        )
        return model

    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            **self.tokenizer_kwargs  # Pass additional keyword arguments
        )
        return tokenizer
    
    
    def load_streamer(self):
        
        streamer = TextStreamer(
            self.tokenizer,
            skip_prompt = True,
            **self.streamer_kwargs)
        
        return streamer
    
    def tokenize(self, prompt : str, **kwargs):
        
        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.cuda()
        
        return input_ids
    
    def infer(self, messages : dict, temperature : float = 0.5, top_p : float = 0.95, top_k : float = 40, max_new_tokens : int = 256, stream : bool = False, remove_prompt : bool = False, add_generation_prompt : bool = False, skip_special_tokens : bool = False, **kwargs):
                        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)



        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512,
            streamer=self.streamer if stream else None
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        
        return response
    
    def get_model_config(self):
        
            config = self.model.config
            
            config_dict = {
            "bos_token_id": str(config.bos_token_id),
            "eos_token_id" : str(config.eos_token_id),
            "pad_token_id" : str(config.pad_token_id),
            "hidden_act" : str(config.hidden_act),
            "hidden_size" : str(config.hidden_size),
            "intermediate_size" : str(config.intermediate_size),
            "max_position_embeddings" : str(config.max_position_embeddings),
            "model_type" : str(config.model_type),
            "num_attention_heads" : str(config.num_attention_heads),
            "num_hidden_layers" : str(config.num_hidden_layers),
            "num_key_value_heads" : str(config.num_key_value_heads),
            "vocab_size" : str(config.vocab_size),
            }
            
            if self.model.config.quantization_config:
                config_dict['quantization_config'] = {
                    "bits" : str(config.quantization_config.bits),
                    "quant_method" : str(config.quantization_config.quant_method),
                    "group_size" : str(config.quantization_config.group_size),
                }

            
            return config_dict