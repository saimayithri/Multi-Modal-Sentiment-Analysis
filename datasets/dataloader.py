from torch.utils.data import DataLoader
from datasets.CMUDataset import CMUData

def getdataloader(dataset_name, batch_size, data_path, num_workers=2):
    """Create train/valid/test dataloaders for the specified dataset.
    
    Args:
        dataset_name: one of 'mosi', 'mosei', 'sims'
        batch_size: batch size for all loaders
        data_path: path to the .pkl data file
        num_workers: number of dataloader workers
        
    Returns:
        dict of dataloaders, tuple of original feature dimensions
    """
    if dataset_name in ['mosi', 'mosei', 'sims']:
        train_set = CMUData(data_path, 'train')
        valid_set = CMUData(data_path, 'valid')
        test_set = CMUData(data_path, 'test')
        
        # Use persistent_workers and pin_memory for better performance
        common_kwargs = {
            'num_workers': num_workers,
            'pin_memory': True,
        }
        # Only use persistent_workers if num_workers > 0
        if num_workers > 0:
            common_kwargs['persistent_workers'] = True

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **common_kwargs)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, **common_kwargs)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, **common_kwargs)
        
        text_dim, audio_dim, vision_dim = train_set.get_dim()
        orig_dim = (text_dim, audio_dim, vision_dim)
        
        return {'train': train_loader, 'valid': valid_loader, 'test': test_loader}, orig_dim
    else:
        raise NotImplementedError(f"Dataloader for {dataset_name} not implemented.")