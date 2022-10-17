import os
import time
import random
import itertools
from typing import List, Tuple
from dataclasses import dataclass

from PIL import Image
import numpy as np
import torch
import sklearn
import scann
import clip
from clip_text_decoder.model import ImageCaptionInferenceModel
from tqdm import tqdm


device = 'cuda' if torch.cuda.is_available() else 'cpu'

PImage = Image.Image


class ClipBase:
    clip_model, clip_preprocess = clip.load('ViT-B/32', device=device)


def load_corpus() -> List[str]:
    paths = [
        'data/ngrams/nouns.txt',
        'data/ngrams/supplemental.txt', # google ngrams missing some words
    ]
    
    with open('data/ngrams/stopwords.txt', 'r') as fin:
        stopwords = set(fin.read().split('\n'))

    def not_contains_stopwords(phrase):
        for word in phrase.split(' '):
            if word in stopwords:
                return False
        return True

    corpus = []
    for path in paths:
        with open(path) as fin:
            phrases = [line.split(',')[0] for line in fin.read().split('\n')][1:]
            phrases = list(filter(not_contains_stopwords, phrases))
            corpus.extend(phrases)
    return list(set(corpus))


class TextNN(ClipBase):
    ResultList = List[Tuple[str, float, int]]

    def __init__(self, topk: int = 10):
        '''
        '''
        self.topk = topk
        self.corpus = load_corpus()
        self.embeddings = self.compute_corpus_embeddings(self.corpus)

        self.search_engine = scann.scann_ops_pybind.builder(
            self.embeddings, 
            topk, 
            'dot_product'
        ).tree(
            num_leaves=2000, 
            num_leaves_to_search=100, 
            training_sample_size=70000
        ).score_ah(
            2, 
            anisotropic_quantization_threshold=0.2
        ).reorder(
            100
        ).build()
    
    def compute_corpus_embeddings(self, corpus: List[str]):
        ''' Returns normalized N x 512 embedding tensor
        '''
        embeddings = clip.tokenize(corpus).to(device)
        with torch.no_grad():
            embeddings = self.clip_model.encode_text(embeddings)
        embeddings /= embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy()

    def search(self, queries: np.array) -> List[ResultList]:
        ''' Return (fidelity score, topk nearest neighbors) for each query embedding

            neighbors: N x k array of top k document indicies, where N is number of queries
        '''
        neighbors, _ = self.search_engine.search_batched(queries)
        results = []
        for query, indicies in zip(queries, neighbors):
            embeddings = self.embeddings[indicies]
            query = np.expand_dims(query, axis=1)
            score = np.sum(embeddings @ query) / self.topk
            results.append((score, [self.corpus[i] for i in indicies]))
        return results

    def prune(self, embeddings: np.array, min_clusters=1, max_clusters=10) -> List[ResultList]:
        ''' Database words seem orthogonal enough that this is not needed
        '''
        return [i for i in range(len(embeddings))]
        # cluster then select top k, alternating between clusters
        '''
        min_loss, min_kmeans = float('inf'), None
        for k in range(min_clusters, max_clusters):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(embeddings)
            if kmeans.inertia_ < min_loss:
                min_loss, min_kmeans = kmeans.inertia_, kmeans

        k = min_kmeans.n_clusters
        '''


class Crop():
    def __init__(self, center: Tuple, size: Tuple, im: PImage):
        self.center = center
        self.size = size
        self.im = im
        self.fidelity_score = None
        self.objects = []

    def distance(self, other) -> float:
        return ((self.center[0] - other.center[0]) ** 2 + (self.center[1] - other.center[1]) ** 2) ** 0.5

    def __repr__(self):
        return f'(Center: {self.center}, Size: {self.size})'


@dataclass
class ImageSegmentationParams:
    distance_cost_multipler : float = 1.0
    num_samples_per_resolution : int = 25
    select_crops_iters : int = 300
    max_crop_size : float = 512
    min_crop_size : float = 512
    fidelity_threshold : float = 0.2


class ImageSegmentationModule(ClipBase):
    def __init__(self):
        self.text_nn = TextNN()
        self.params = ImageSegmentationParams()

    def __call__(self, im: PImage, num_pruned_per_resolution: List[int], scale_factors: List[float]) -> List[Crop]:
        ret_crops = []
        for num_samples, factor in zip(num_pruned_per_resolution, scale_factors):
            im_scaled = self.scale(im, factor)
            print(f'Generating crops for image size: {im_scaled.size}...')
            width, height = im_scaled.size
            if width < self.params.max_crop_size or height < self.params.max_crop_size:
                break
            crops = self.generate_crops(im_scaled)
            crops = self.select_crops(crops, num_samples)
            ret_crops.extend(crops)
        return ret_crops

    def generate_crops(self, im: PImage) -> List[Crop]:
        print('Generating Crops...')

        im_width, im_height = im.size
        crops = []
        for i in range(self.params.num_samples_per_resolution):
            half_width, half_height = \
                random.randint(self.params.min_crop_size // 2,   # image/scenes are wider than taller
                               self.params.max_crop_size // 2), \
                random.randint(self.params.min_crop_size // 2, 
                               self.params.max_crop_size // 2)

            x = random.randint(half_width, im_width - half_width)
            y = random.randint(half_height, im_height - half_height)
            new_im = im.crop((x - half_width, y - half_height, 
                              x + half_width, y + half_height))

            crops.append(Crop((x, y), new_im.size, new_im))

        results = self.compute_fidelity_scores(crops)
        ret_crops = []
        for i, crop in enumerate(crops):
            if results[i][0] > self.params.fidelity_threshold:
                ret_crops.append(crop)
        return ret_crops

    def select_crops(self, crops: List[PImage], num_samples: int) -> List[Crop]:
        print('Pruning Crops...')

        def cost_fn(_crops) -> float:
            ''' Negative all pairs distance sum '''
            cost = 0
            for c1 in _crops:
                for c2 in _crops:
                    cost += c1.distance(c2) * self.params.distance_cost_multipler
            return -cost   

        min_cost, min_crops = float('inf'), None
        for i in range(self.params.select_crops_iters):
            selected_crops = random.sample(crops, num_samples)
            cost = cost_fn(selected_crops)
            if cost < min_cost:
                min_cost, min_crops = cost, selected_crops
        return min_crops

    def compute_fidelity_scores(self, crops: List[Crop]) -> float:
        print('Computing Crop Fidelity Scores')

        embeddings = []
        for crop in tqdm(crops):
            im = self.clip_preprocess(crop.im).unsqueeze(0).to(device)
            embeddings.append(self.clip_model.encode_image(im))
        embeddings = torch.cat(embeddings, dim=0)
        embeddings /= embeddings.norm(dim=-1, keepdim=True)

        results = self.text_nn.search(embeddings.detach().cpu().numpy())
        print(f'{len(results)} results: ')
        print('\n'.join(map(str, sorted(results, reverse=True)[:10])))
        print('...')

        for crop, (score, objects) in zip(crops, results):
            crop.fidelity_score = score
            crop.objects = objects

        return results

    def scale(self, im: PImage, factor: float) -> PImage:
        width, height = im.size
        return im.resize((int(width * factor), int(height * factor)))


class CaptionModule(ClipBase):
    def __init__(self, path: str, originial_image_size: Tuple[int, int]):
        self.clip_caption_model = ImageCaptionInferenceModel.load(path).to(device)
        self.originial_image_size = originial_image_size

    def __call__(self, crops: List[Crop], beam_size=1, relational=True):
        print('Generating captions...')

        make_caption = lambda crop : self.clip_caption_model(crop.im, beam_size=beam_size).lower()
        captions = list(map(make_caption, crops))
        if relational:
            captions = [self.point_to_positional_desc(crop.center) + ' ' + ', '.join(crop.objects[:4]) + '. And ' + text # clip doesn't know what iPad is
                for text, crop in zip(captions, crops)]
        return captions

    POSITIONAL_DESC = list(itertools.product(['Top', 'Center', 'Bottom'], ['left', 'mid', 'right']))

    def point_to_positional_desc(self, point: Tuple[int, int]):
        width, height = self.originial_image_size
        x, y = point
        x_quadrant = x // (width // 3)
        y_quadrant = y // (height // 3)
        pos = self.POSITIONAL_DESC[y_quadrant * 3 + x_quadrant]
        return str(pos[0]) + ' ' + pos[1] + ' there is '


if __name__ == '__main__':
    print('Running clip engine...')
    while True:
        try:
            im = Image.open('dock/input_frame.png')
            print('Processing image')
        except:
            time.sleep(1)
            continue
        
        time.sleep(2)
        os.remove('dock/input_frame.png')

        crops = ImageSegmentationModule()(
            im,
            num_pruned_per_resolution=[2, 1],
            scale_factors=[0.625, 0.5]
        )
        print(len(crops))
        captions = CaptionModule(
            'models/clip_text_decoder.pt', im.size
        )(
            crops,
            beam_size=3,
            relational=True
        )
        captions = sorted(captions)
        captions = [c.strip('.').replace('  ', ' ') for c in captions]
        captions = list(set(captions))

        print(captions)

        with open('dock/output.txt', 'w') as fout:
            for caption in captions:
                fout.write(caption + '\n')

        time.sleep(10)
        print('Waiting for next image...')
       #os.remove('dock/output.txt')



