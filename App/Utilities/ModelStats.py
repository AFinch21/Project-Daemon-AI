import psutil
import torch


class ModelStatistics():
    def __init__(self, model_id):
        self.model_id = model_id
        
        
    def get_ram(self):
        mem = psutil.virtual_memory()
        free = mem.available / 1024 ** 3
        total = mem.total / 1024 ** 3
        used = total - free
        total_cubes = 24
        free_cubes = int(total_cubes * free / total)
        
        graphic = f'RAM: {total - free:.2f}/{total:.2f}GB\t RAM:[' + (total_cubes - free_cubes) * '▮' + free_cubes * '▯' + ']'
        
        
        
        return graphic, total, free, used


    def get_vram(self):
        free = torch.cuda.mem_get_info()[0] / 1024 ** 3
        total = torch.cuda.mem_get_info()[1] / 1024 ** 3
        used = total - free
        total_cubes = 24
        free_cubes = int(total_cubes * free / total)

        graphic = f'VRAM: {total - free:.2f}/{total:.2f}GB\t VRAM:[' + (
                total_cubes - free_cubes) * '▮' + free_cubes * '▯' + ']'
        return graphic, total, free, used
        
    def get_model_id(self):
        
        return self.model_id
    
    