from huggingface_hub import HfApi

class HFManager():
    
    def __init__(self):
        self.api = HfApi()
        
    def search_models(self, author = None, sort_by = 'downloads', n_results = 5):
        
        models = self.api.list_models(
            author=author,
            task="text-generation",
            sort=sort_by, 
            direction=-1, 
            limit=n_results)
        
        model_list = []
        
        for model in models:
            model_list.append(model.id)
        
        print('MODELS LIST IN CLASS:', model_list)
        
        return model_list