import torch
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import find_classes, make_dataset
import torchvision.transforms as transforms
import transforms as T
import os

# reference: https://github.com/pytorch/vision/blob/b3b74481a113610b33d27ab703733dad6d79bc91/torchvision/datasets/kinetics.py#L142
class Kinetics(VisionDataset):
    def __init__(self,
                 root: str,
                 frames_per_clip: int,
                 num_classes: int,
                 frame_rate: int,
                 split: str = 'train',
                 step_between_clips: int = 1,
                 output_format: str = "THWC",
                 transform = None,
                 extensions = tuple("mp4"),
               ):
        
        
        self.split_folder = os.path.join(root, split)
        self.classes, self.class_to_idx = find_classes(self.split_folder)
        print(self.class_to_idx)
        self.samples = make_dataset(self.split_folder, self.class_to_idx, extensions, is_valid_file=None)
        video_list = [x[0] for x in self.samples]
        self.video_clips = VideoClips(
            video_list,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            # output_format=output_format,
        )
        self.transform = transform
    
    def __len__(self):
        return self.video_clips.num_clips()
    
    def __getitem__(self, idx: int):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        t,h,w,c = video.shape
        assert t!=0 and h!=0 and w!=0 and c!=0
        if t < 50:
            fill_n = 50 - t
            video = torch.cat([video, torch.zeros((fill_n, h, w, c), dtype=torch.uint8)], dim=0)
        label = self.samples[video_idx][1]

        if self.transform is not None:
            video = self.transform(video)

        # return video, audio, label
        return video, label

    def get_classes(self):
        return self.class_to_idx
    
if __name__ == '__main__':
    root='/home/workspaces/datasets/ados_act/ados/'
    split = 'mix_2/'
    dirs = os.listdir(os.path.join(root, split))
    num_classes = len(dirs)
    if '.ipynb_checkpoints' in dirs:
        os.rmdir(os.path.join(root, split, '.ipynb_checkpoints'))
        num_classes -= 1
    
    transform = transforms.Compose([  
                         T.ToFloatTensorInZeroOne(),
                         T.Resize((240, 320)), #200 260  -> 240 320 TODO
                         T.RandomHorizontalFlip(),
                         #T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
                         T.RandomCrop((224, 224))]) #172 224
    
    dataset = Kinetics(root=root, split=split, frames_per_clip=50, num_classes=num_classes, frame_rate=5, transform=transform)