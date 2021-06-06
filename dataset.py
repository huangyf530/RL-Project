from torch.utils.data import Dataset

def collate_fn(batch):
    genres = []
    keywords = []
    poems = []
    for genre, keyword, poem in batch:
        genres.append(int(genre))
        keywords.append(keyword)
        poems.append(poem)
    return genres, keywords, poems

class CustomDataset(Dataset):
    def __init__(self, filename):
        self.lines = []
        with open(filename) as f:
            for linenum, line in enumerate(f):
                line = line.strip()
                if line == '':
                    continue
                self.lines.append(line.split('#'))
    
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, index):
        genre, keywords, poem = self.lines[index]
        keywords = keywords.split('|')
        return genre, keywords, poem