import os
import numpy as np
from sklearn.decomposition import IncrementalPCA
import rasterio
import glob

def process_crism_dataset(input_dirs, output_dir, n_components=10, batch_size=1000):
    """
    Process CRISM multispectral tiles using incremental PCA.
    
    Parameters:
    -----------
    input_dirs : list
        List of directories containing the CRISM tiles
    output_dir : str
        Directory to save the transformed tiles
    n_components : int
        Number of principal components to keep
    batch_size : int
        Number of pixels to process at once
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the incremental PCA
    ipca = IncrementalPCA(n_components=n_components)
    
    # Phase 1: Fit the PCA incrementally
    print("Phase 1: Calculating PCA transform...")
    
    # Find all tiles across all directories
    all_tiles = []
    for directory in input_dirs:
        # Adjust the file pattern based on the actual file extensions
        pattern = os.path.join(directory, "*.tif")  # Modify extension as needed
        tiles = glob.glob(pattern)
        all_tiles.extend(tiles)
    
    print(f"Found {len(all_tiles)} tiles to process")
    
    # Process each tile to fit the PCA
    for i, tile_path in enumerate(all_tiles):
        print(f"Processing tile {i+1}/{len(all_tiles)}: {os.path.basename(tile_path)}")
        
        with rasterio.open(tile_path) as src:
            # Get metadata for later use
            profile = src.profile
            
            # Read data and reshape for PCA
            # Assuming all bands should be used for PCA
            data = src.read()
            
            # Reshape to [n_samples, n_features]
            # n_samples = pixels, n_features = bands
            n_bands, height, width = data.shape
            data_reshaped = data.reshape(n_bands, height * width).T
            
            # Filter out NaN or NoData values if present
            valid_mask = ~np.isnan(data_reshaped).any(axis=1)
            valid_data = data_reshaped[valid_mask]
            
            # Process in batches to save memory
            if len(valid_data) > batch_size:
                for i in range(0, len(valid_data), batch_size):
                    batch = valid_data[i:i+batch_size]
                    ipca.partial_fit(batch)
            else:
                ipca.partial_fit(valid_data)
    
    print("PCA transform calculated!")
    
    # Save the PCA model for future use
    pca_path = os.path.join(output_dir, "crism_pca_model.npz")
    np.savez(pca_path, 
             mean=ipca.mean_,
             components=ipca.components_,
             explained_variance=ipca.explained_variance_,
             explained_variance_ratio=ipca.explained_variance_ratio_,
             n_components=ipca.n_components_)
    
    # Phase 2: Apply the PCA transform to each tile
    print("Phase 2: Applying PCA transform to each tile...")
    
    for i, tile_path in enumerate(all_tiles):
        print(f"Transforming tile {i+1}/{len(all_tiles)}: {os.path.basename(tile_path)}")
        
        # Generate output file path
        tile_name = os.path.basename(tile_path)
        output_path = os.path.join(output_dir, f"pca_{tile_name}")
        
        with rasterio.open(tile_path) as src:
            profile = src.profile
            
            # Update profile for the output file
            profile.update(
                count=n_components,  # Number of bands is now n_components
                dtype=np.float32
            )
            
            # Read data
            data = src.read()
            n_bands, height, width = data.shape
            
            # Initialize output array
            transformed = np.zeros((n_components, height, width), dtype=np.float32)
            
            # Process in chunks to save memory
            for row_start in range(0, height, 100):  # Process 100 rows at a time
                row_end = min(row_start + 100, height)
                
                # Extract chunk
                chunk = data[:, row_start:row_end, :]
                
                # Reshape to [n_samples, n_features]
                chunk_reshaped = chunk.reshape(n_bands, (row_end - row_start) * width).T
                
                # Create mask for invalid values
                valid_mask = ~np.isnan(chunk_reshaped).any(axis=1)
                
                # Initialize transformed chunk
                transformed_chunk = np.zeros((chunk_reshaped.shape[0], n_components))
                
                # Transform valid data
                transformed_chunk[valid_mask] = ipca.transform(chunk_reshaped[valid_mask])
                
                # Reshape back to spatial dimensions
                transformed_chunk = transformed_chunk.T.reshape(n_components, row_end - row_start, width)
                
                # Store in output array
                transformed[:, row_start:row_end, :] = transformed_chunk
            
            # Write transformed data
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(transformed)
    
    print("PCA transformation complete!")
    return pca_path

# Example usage
if __name__ == "__main__":
    # List directories containing CRISM tiles
    input_directories = ["/mnt/nili_e/Software/fractal_complexity/data/Jezero_HiRISE_complexity_tiles"]
    
    output_directory = "/mnt/nili_e/Software/fractal_complexity/data/Jezero_HiRISE_complexity_pca_tiles"
    
    # Run the incremental PCA process
    process_crism_dataset(input_directories, output_directory, n_components=9)