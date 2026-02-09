from colorama import Fore
from .image_reg import *

class AFLWReg(ImageRegression):
    def get_notice(self):
        notice = f'{Fore.YELLOW}' \
                 f'\nAFLW Dataset: ' \
                 f'\n    Nose, eyes and ears: Head pose estimation by locating facial keypoints, ' \
                 f'\n        Aryaman Gupta, Kalpit Thakkar, Vineet Gandhi, P J Narayanan, ICASSP, 2019, ' \
                 f'\n        https://arxiv.org/abs/1812.00739 ' \
                 f'{Fore.RESET}\n'
                 
    def get_dataset_info(self):
        return None
    
    