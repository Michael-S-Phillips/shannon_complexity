import rasterio
from rasterio.windows import Window
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.util import view_as_windows
from skimage.measure import shannon_entropy
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil

class tiledComplexity:
    """
    Class to calculate Shannon entropy of image tiles.
    
    Parameters:
    -----------
    img_path : str
        Path to the image file
    tile_size : int
        Size of the tiles to be created
    overlap : int
        Overlap between tiles
    kernel_sizes : list
        List of kernel sizes for entropy calculation
    out_dir : str
        Directory to save the output complexity maps
    crop : bool
        Whether to crop the output to the original tile size
    batch_size : int
        Number of tiles to process in a batch
    """
    def __init__(self, img_path, tile_size, overlap, kernel_sizes, out_dir=None, crop=False, batch_size=None, debug=False):
        self.img_path = img_path
        self.tile_size = tile_size
        self.overlap = overlap
        self.kernel_sizes = kernel_sizes
        self.out_dir = out_dir
        self.crop = crop
        self.debug = debug
        
        # Auto-determine batch size based on available memory if not provided
        if batch_size is None:
            # Use 50% of available memory for safe processing
            available_memory = psutil.virtual_memory().available
            # Rough estimate: each float32 takes 4 bytes, multiply by dimensions
            est_memory_per_tile = 4 * tile_size * tile_size * len(kernel_sizes) * 3  # 3x for safety
            self.batch_size = max(1, int(available_memory * 0.5 / est_memory_per_tile))
        else:
            self.batch_size = batch_size
            
        # Initialize image dimensions
        with rasterio.open(self.img_path) as src:
            self.img_width = src.width
            self.img_height = src.height
            self.src_transform = src.transform
            self.src_crs = src.crs

    def visualize_tiles(self, save_path=None):
        """
        Create a visualization of how tiles will be created and cropped.
        Useful for debugging tile alignment issues.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the visualization. If None, will display the plot
        """
        with rasterio.open(self.img_path) as src:
            # Get a small sample of the image for visualization
            sample_size = min(1024, min(self.img_width, self.img_height))
            sample = src.read(1, window=Window(0, 0, sample_size, sample_size))
        
        # Create a figure to visualize the tiling
        plt.figure(figsize=(12, 10))
        plt.imshow(sample, cmap='gray')
        plt.title(f"Tile Boundaries (tile_size={self.tile_size}, overlap={self.overlap})")
        
        # Get step size (distance between tile starts)
        step = self.tile_size - self.overlap
        
        # Plot tile boundaries
        for i in range(0, sample_size, step):
            for j in range(0, sample_size, step):
                # Full tile with overlap
                rect = plt.Rectangle((j, i), self.tile_size, self.tile_size, 
                                    fill=False, edgecolor='red', linestyle='-', linewidth=1)
                plt.gca().add_patch(rect)
                
                # Cropped tile (if crop=True)
                if self.crop and self.overlap > 0:
                    overlap_half = self.overlap // 2
                    
                    # Determine if this is an edge
                    is_left = j == 0
                    is_top = i == 0
                    
                    # Calculate crop coordinates
                    crop_left = j if is_left else j + overlap_half
                    crop_top = i if is_top else i + overlap_half
                    crop_right = min(j + self.tile_size, sample_size)
                    crop_bottom = min(i + self.tile_size, sample_size)
                    
                    if not is_left and crop_right > j + step:
                        crop_right = j + step
                    if not is_top and crop_bottom > i + step:
                        crop_bottom = i + step
                    
                    # Draw the cropped region
                    crop_rect = plt.Rectangle((crop_left, crop_top), 
                                            crop_right - crop_left, 
                                            crop_bottom - crop_top,
                                            fill=False, edgecolor='blue', linestyle='-', linewidth=2)
                    plt.gca().add_patch(crop_rect)
        
        plt.xlim(0, sample_size)
        plt.ylim(sample_size, 0)  # Invert y-axis for proper image display
        plt.grid(True, linestyle='--', alpha=0.5)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
    
    def process(self, start_index=0):
        """Process the image in batches using parallel processing."""
        # Get the list of windows (tiles)
        windows, window_transforms = self.get_windows()
        total_windows = len(windows)
        
        print(f"Processing {total_windows} tiles in batches of {self.batch_size}")
        
        # Process in batches to conserve memory
        results = []
        with ProcessPoolExecutor() as executor:
            num_workers = executor._max_workers
            print(f"Using {num_workers} CPU workers for processing.")
            
            # Process tiles in batches
            for i in range(start_index, total_windows, self.batch_size):
                batch_windows = windows[i:min(i+self.batch_size, total_windows)]
                print(f"Processing batch {i//self.batch_size + 1}/{(total_windows-1)//self.batch_size + 1} ({len(batch_windows)} tiles)")
                
                # Submit batch for processing
                futures = [executor.submit(self.kernel_entropy, window) for window in batch_windows]
                
                # Use as_completed to process results as they finish
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tiles"):
                    try:
                        result = future.result()
                        if result is not None:  # Only append if not saving to disk
                            results.append(result)
                    except Exception as e:
                        print(f"Error processing tile: {e}")
        
        return results if not self.out_dir else None

    def get_windows(self):
        """Get the list of windows (tiles) for processing."""
        window_list = []
        window_transform_list = []
        
        print(f"Image dimensions: {self.img_width}x{self.img_height}")
        
        # Loop through the image and create tiles
        for i in range(0, self.img_height, self.tile_size - self.overlap):
            for j in range(0, self.img_width, self.tile_size - self.overlap):
                # Define the window for the current tile
                window = Window(j, i, min(self.tile_size, self.img_width - j), 
                                    min(self.tile_size, self.img_height - i))
                
                window_transform = rasterio.windows.transform(window, self.src_transform)
                window_list.append(window)
                window_transform_list.append(window_transform)
        
        return window_list, window_transform_list

    def get_tile_window(self, window):
        """Get a tile from the image using a specified window."""
        with rasterio.open(self.img_path) as src:
            # Read the tile
            tile = src.read(1, window=window)
            window_transform = rasterio.windows.transform(window, src.transform)
            tile_extent = src.window_bounds(window)
            tile_crs = src.crs
        
        return tile, self.src_transform, window_transform, tile_extent, tile_crs

    def optimized_shannon_entropy(self, img, kernel_size):
        """
        Optimized Shannon entropy calculation using a histogram-based approach
        when possible for smaller kernel sizes.
        """
        if kernel_size <= 7:  # For small kernels, use histogram-based approach
            padded_map = np.pad(img, pad_width=kernel_size // 2, mode='reflect')
            kernels = view_as_windows(padded_map, (kernel_size, kernel_size))
            
            # Pre-allocate output array
            entropy_map = np.zeros(img.shape, dtype=np.float32)
            
            # Get unique values range for histogram bins
            # For 8-bit imagery, 0-255 is common
            unique_vals = np.unique(img)
            min_val, max_val = unique_vals.min(), unique_vals.max()
            bins = max(min(256, len(unique_vals)), 10)  # Reasonable bin count
            
            # Process each window
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    kernel = kernels[i, j]
                    hist, _ = np.histogram(kernel, bins=bins, range=(min_val, max_val), density=True)
                    # Remove zeros to avoid log(0)
                    hist = hist[hist > 0]
                    entropy_map[i, j] = -np.sum(hist * np.log2(hist))
            
            return entropy_map
        else:
            # For larger kernels, use the standard approach
            padded_map = np.pad(img, pad_width=kernel_size // 2, mode='reflect')
            kernels = view_as_windows(padded_map, (kernel_size, kernel_size))
            
            # Pre-allocate output array
            entropy_map = np.zeros(img.shape, dtype=np.float32)
            
            # Process each window
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    entropy_map[i, j] = shannon_entropy(kernels[i, j])
            
            return entropy_map

    def kernel_entropy(self, window):
        """Process a single window (tile) to calculate complexity maps."""
        base_img, src_transform, window_transform, tile_extent, tile_crs = self.get_tile_window(window)
        
        # Skip if the window is too small (edge cases)
        if base_img.shape[0] < 2 or base_img.shape[1] < 2:
            print(f"Skipping window at {window.col_off},{window.row_off} - too small: {base_img.shape}")
            return None
            
        complexity_map_cube = np.zeros((base_img.shape[0], base_img.shape[1], len(self.kernel_sizes)), dtype=np.float32)

        for b, kernel_size in enumerate(self.kernel_sizes):
            # Skip if kernel size is larger than the image
            if kernel_size > min(base_img.shape):
                print(f"Skipping kernel size {kernel_size} for window at {window.col_off},{window.row_off} - too large")
                complexity_map_cube[:, :, b] = 0
                continue
                
            # Calculate entropy map
            kernel_complexity_map = self.optimized_shannon_entropy(base_img, kernel_size)
            complexity_map_cube[:, :, b] = kernel_complexity_map

        if self.out_dir is not None:
            # Save the complexity map as a GeoTIFF
            output_path = os.path.join(self.out_dir, f"complexity_map_{window.col_off}_{window.row_off}.tif")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # No-overlap cropping logic - removes overlap regions completely
            if self.crop and self.overlap > 0:
                # Get original shape for reference
                original_height, original_width = complexity_map_cube.shape[0], complexity_map_cube.shape[1]
                
                # Determine if this is an edge tile
                is_left_edge = window.col_off == 0
                is_top_edge = window.row_off == 0
                is_right_edge = window.col_off + window.width >= self.img_width
                is_bottom_edge = window.row_off + window.height >= self.img_height
                
                # For a tile with overlap=4, we want to remove 2 pixels from each side
                # (except at image edges)
                overlap_half = self.overlap // 2
                
                # Set crop indices to completely remove overlap regions
                # This ensures tiles align perfectly with no overlap
                left_crop = 0 if is_left_edge else overlap_half
                top_crop = 0 if is_top_edge else overlap_half
                right_crop = original_width if is_right_edge else original_width - overlap_half
                bottom_crop = original_height if is_bottom_edge else original_height - overlap_half
                
                # Visual explanation of cropping:
                # For a tile with overlap=4 (overlap_half=2):
                #
                # Original tile with overlap:
                # +------------------------+
                # |                        |
                # |     tile data          |
                # |                        |
                # +------------------------+
                #
                # After cropping:
                # +------------------------+
                # |  X  |              |  X|
                # |-----+              +---|
                # |     |              |   |
                # |     |  final tile  |   |
                # |     |              |   |
                # |-----+              +---|
                # |  X  |              |  X|
                # +------------------------+
                # X = removed overlap regions
                
                # Crop the complexity map to remove all overlap regions
                complexity_map_cube = complexity_map_cube[top_crop:bottom_crop, left_crop:right_crop, :]
                
                # Update the window transform to account for cropping
                window_transform = rasterio.windows.transform(
                    Window(
                        window.col_off + left_crop,
                        window.row_off + top_crop,
                        right_crop - left_crop,
                        bottom_crop - top_crop
                    ),
                    src_transform
                )
                
                # Debug information
                if self.debug:
                    print(f"Tile at ({window.col_off}, {window.row_off}): Original size={original_width}x{original_height}, "
                          f"Cropped size={right_crop-left_crop}x{bottom_crop-top_crop}")
            
            # Write the output file
            with rasterio.open(output_path, 'w', driver='GTiff', height=complexity_map_cube.shape[0],
                            width=complexity_map_cube.shape[1], count=len(self.kernel_sizes),
                            dtype=complexity_map_cube.dtype, crs=tile_crs,
                            transform=window_transform) as dst:
                for b in range(len(self.kernel_sizes)):
                    dst.write(complexity_map_cube[:, :, b], b + 1)
            
            return None  # Return None when saving to disk to save memory
        else:
            return {
                'window': window,
                'complexity_map': complexity_map_cube,
                'transform': window_transform,
                'extent': tile_extent
            }
    
    def create_mosaic(self, output_path=None, max_memory_usage_gb=4):
        """
        Assemble all output tiles into a single mosaic complexity cube.
        
        Parameters:
        -----------
        output_path : str, optional
            Path to save the output mosaic. If None, will use 'complexity_mosaic.tif' in the output directory.
        max_memory_usage_gb : float, optional
            Maximum memory usage in GB for the mosaic creation. Larger values may speed up processing
            but require more RAM.
            
        Returns:
        --------
        str : Path to the created mosaic file
        """
        import os
        import glob
        import rasterio
        from rasterio.merge import merge
        import math
        import numpy as np
        from tqdm import tqdm
        import gc  # Add garbage collection
        
        if self.out_dir is None:
            raise ValueError("Cannot create mosaic: No output directory was specified during initialization")
        
        if output_path is None:
            output_path = os.path.join(self.out_dir, "complexity_mosaic.tif")
        
        # Find all tile files in the output directory
        search_pattern = os.path.join(self.out_dir, "complexity_map_*.tif")
        tile_files = glob.glob(search_pattern)
        
        if not tile_files:
            raise FileNotFoundError(f"No tile files found in {self.out_dir} matching pattern 'complexity_map_*.tif'")
        
        print(f"Found {len(tile_files)} tile files to merge")
        
        # Get metadata from the first tile to determine dimensions
        with rasterio.open(tile_files[0]) as src:
            meta = src.meta.copy()
            num_bands = src.count  # Number of kernel sizes
        
        # Calculate number of tiles to process in each batch based on memory constraints
        # Rough estimation: float32 = 4 bytes per pixel
        bytes_per_pixel = 4  # float32
        # Estimate average tile size from first tile
        with rasterio.open(tile_files[0]) as sample:
            avg_tile_height = sample.height
            avg_tile_width = sample.width
        
        # Calculate memory usage per tile (in bytes)
        tile_memory_usage = avg_tile_height * avg_tile_width * bytes_per_pixel  # Just for one band
        
        # Calculate max tiles per batch (more conservative estimation)
        max_memory_bytes = max_memory_usage_gb * 1024 * 1024 * 1024 * 0.7  # Using only 70% of specified memory
        max_tiles_per_batch = max(1, int(max_memory_bytes / tile_memory_usage))
        
        print(f"Processing tiles in batches of {max_tiles_per_batch} to fit within {max_memory_usage_gb}GB memory limit")
        
        # Create an initial empty output file to write band by band
        # First, determine the bounds of the final mosaic
        print("Calculating final mosaic bounds...")
        merged_bounds = None
        for file_path in tqdm(tile_files[:min(100, len(tile_files))], desc="Sampling bounds"):  # Sample a subset
            with rasterio.open(file_path) as src:
                bounds = rasterio.transform.array_bounds(src.height, src.width, src.transform)
                if merged_bounds is None:
                    merged_bounds = list(bounds)
                else:
                    merged_bounds[0] = min(merged_bounds[0], bounds[0])  # min bottom
                    merged_bounds[1] = min(merged_bounds[1], bounds[1])  # min left
                    merged_bounds[2] = max(merged_bounds[2], bounds[2])  # max top
                    merged_bounds[3] = max(merged_bounds[3], bounds[3])  # max right
        
        # Calculate dimensions and transform for the final mosaic
        with rasterio.open(tile_files[0]) as src:
            res = src.res
            final_width = int((merged_bounds[3] - merged_bounds[1]) / res[0])
            final_height = int((merged_bounds[2] - merged_bounds[0]) / res[1])
            final_transform = rasterio.transform.from_bounds(
                merged_bounds[1], merged_bounds[0], 
                merged_bounds[3], merged_bounds[2], 
                final_width, final_height
            )
        
        # Update metadata for the output file
        meta.update({
            'driver': 'GTiff',
            'height': final_height,
            'width': final_width,
            'count': num_bands,
            'transform': final_transform,
            'tiled': True,
            'compress': 'lzw',  # Add compression to reduce file size
            'blockxsize': 256,  # Optimal block size for most operations
            'blockysize': 256
        })
        
        # Create the output file
        with rasterio.open(output_path, 'w', **meta) as dst:
            # Process one band at a time
            for band in range(1, num_bands + 1):
                print(f"Processing band {band} of {num_bands} (kernel size: {self.kernel_sizes[band-1]})")
                
                # Process tiles in batches to manage memory
                num_batches = math.ceil(len(tile_files) / max_tiles_per_batch)
                final_band_data = np.zeros((final_height, final_width), dtype=meta['dtype'])
                final_band_data.fill(meta.get('nodata', 0))  # Fill with nodata value if available, or 0
                
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * max_tiles_per_batch
                    end_idx = min((batch_idx + 1) * max_tiles_per_batch, len(tile_files))
                    batch_files = tile_files[start_idx:end_idx]
                    
                    print(f"  Processing batch {batch_idx+1}/{num_batches} ({len(batch_files)} tiles)")
                    
                    # Open all files in the batch for this band
                    batch_sources = []
                    for file_path in tqdm(batch_files, desc="Opening tiles"):
                        try:
                            with rasterio.open(file_path) as src:
                                # Read only the current band
                                band_data = src.read(band)
                                batch_sources.append((band_data, src.transform))
                        except Exception as e:
                            print(f"Error opening {file_path}: {e}")
                    
                    # Merge the batch
                    if batch_sources:
                        try:
                            batch_merged, batch_transform = self.merge_arrays(batch_sources)
                            
                            # Write the batch data to the appropriate location in the final band array
                            self.write_array_to_position(
                                batch_merged, batch_transform,
                                final_band_data, final_transform
                            )
                        except Exception as e:
                            print(f"Error merging batch: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    # Clear batch data to free memory
                    batch_sources = None
                    gc.collect()  # Force garbage collection
                
                # Write the completed band to the file
                dst.write(final_band_data, band)
                dst.set_band_description(band, f"Complexity (kernel={self.kernel_sizes[band-1]})")
                
                # Clear band data to free memory before processing next band
                final_band_data = None
                gc.collect()  # Force garbage collection
        
        print(f"Mosaic successfully created: {output_path}")
        print(f"Dimensions: {final_width}x{final_height}, {num_bands} bands")
        
        return output_path

    def write_array_to_position(self, source_array, source_transform, target_array, target_transform):
        """
        Write a source array into the correct position within a larger target array
        based on their respective transforms.
        
        Parameters:
        -----------
        source_array : numpy.ndarray
            The source data array
        source_transform : affine.Affine
            The transform of the source array
        target_array : numpy.ndarray
            The target data array to write into
        target_transform : affine.Affine
            The transform of the target array
        """
        import numpy as np
        import rasterio
        from rasterio.windows import from_bounds
        
        # Get the bounds of the source array
        source_height, source_width = source_array.shape
        source_bounds = rasterio.transform.array_bounds(
            source_height, source_width, source_transform)
        
        # Calculate the window in the target array that corresponds to these bounds
        window = from_bounds(
            source_bounds[1], source_bounds[0], 
            source_bounds[3], source_bounds[2],
            target_transform
        )
        
        # Convert window to integer pixel coordinates
        col_start = max(0, int(window.col_off))
        row_start = max(0, int(window.row_off))
        col_end = min(target_array.shape[1], int(window.col_off + window.width))
        row_end = min(target_array.shape[0], int(window.row_off + window.height))
        
        # Calculate source array slice coordinates
        # This assumes that pixel sizes match between source and target
        # For perfect accuracy, resampling might be needed
        s_col_start = max(0, -int(window.col_off))
        s_row_start = max(0, -int(window.row_off))
        s_col_end = s_col_start + (col_end - col_start)
        s_row_end = s_row_start + (row_end - row_start)
        
        # Ensure we don't go beyond the source array dimensions
        s_col_end = min(source_width, s_col_end)
        s_row_end = min(source_height, s_row_end)
        
        # Adjust target window if source is smaller
        if s_col_end - s_col_start < col_end - col_start:
            col_end = col_start + (s_col_end - s_col_start)
        if s_row_end - s_row_start < row_end - row_start:
            row_end = row_start + (s_row_end - s_row_start)
        
        # Make sure we're not trying to write outside the target array
        if (col_start >= target_array.shape[1] or row_start >= target_array.shape[0] or
                col_end <= 0 or row_end <= 0):
            return  # Nothing to write
            
        # Copy data from source to target
        target_array[row_start:row_end, col_start:col_end] = \
            source_array[s_row_start:s_row_end, s_col_start:s_col_end]
    
    def merge_arrays(self, datasets):
        """
        Merge multiple arrays with their transforms into a single array.
        
        Parameters:
        -----------
        datasets : list of tuples
            List of (data, transform) tuples
        
        Returns:
        --------
        tuple : (merged_data, merged_transform)
        """
        import numpy as np
        from rasterio.merge import merge
        import gc  # For garbage collection
        
        # Create a list to store the MemoryFile objects so they stay in scope
        memfiles = []
        rasterio_datasets = []
        
        try:
            # Convert datasets format to what rasterio.merge expects
            for idx, (data, transform) in enumerate(datasets):
                # Skip empty arrays
                if data.size == 0:
                    continue
                    
                # Create a MemoryFile-based dataset with the array data
                from rasterio.io import MemoryFile
                memfile = MemoryFile()
                memfiles.append(memfile)  # Keep reference to prevent garbage collection
                
                # Make sure we're using the right data type - float32 is usually sufficient
                if data.dtype != np.float32 and data.dtype != np.uint8:
                    data = data.astype(np.float32)
                
                # Set nodata value if not present
                nodata = 0.0 if data.dtype == np.float32 else 0
                
                with memfile.open(
                    driver='GTiff',
                    height=data.shape[0],
                    width=data.shape[1],
                    count=1,
                    dtype=data.dtype,
                    transform=transform,
                    crs=None,
                    nodata=nodata
                ) as dataset:
                    dataset.write(data, 1)
                
                # Get a copy of the dataset from the MemoryFile
                rasterio_datasets.append(memfile.open())
            
            # Skip the merge if no valid datasets
            if not rasterio_datasets:
                return np.array([]), None
                
            # Merge the datasets with proper error handling
            try:
                merged_data, merged_transform = merge(
                    rasterio_datasets,
                    nodata=-9999.0,  # Explicitly set nodata value
                    method='first'   # Use 'first' method for faster processing
                )
                return merged_data[0], merged_transform
            except Exception as e:
                print(f"Error during merge operation: {e}")
                import traceback
                traceback.print_exc()
                return np.array([]), None
        finally:
            # Close all resources
            for dataset in rasterio_datasets:
                try:
                    dataset.close()
                except:
                    pass
                    
            for memfile in memfiles:
                try:
                    memfile.close()
                except:
                    pass
            
            # Force garbage collection
            rasterio_datasets = None
            memfiles = None
            gc.collect()
            
    # def create_mosaic(self, output_path=None, max_memory_usage_gb=4):
    #     """
    #     Assemble all output tiles into a single mosaic complexity cube.
        
    #     Parameters:
    #     -----------
    #     output_path : str, optional
    #         Path to save the output mosaic. If None, will use 'complexity_mosaic.tif' in the output directory.
    #     max_memory_usage_gb : float, optional
    #         Maximum memory usage in GB for the mosaic creation. Larger values may speed up processing
    #         but require more RAM.
            
    #     Returns:
    #     --------
    #     str : Path to the created mosaic file
    #     """
    #     import os
    #     import glob
    #     import rasterio
    #     from rasterio.merge import merge
    #     import math
    #     import numpy as np
    #     from tqdm import tqdm
        
    #     if self.out_dir is None:
    #         raise ValueError("Cannot create mosaic: No output directory was specified during initialization")
        
    #     if output_path is None:
    #         output_path = os.path.join(self.out_dir, "complexity_mosaic.tif")
        
    #     # Find all tile files in the output directory
    #     search_pattern = os.path.join(self.out_dir, "complexity_map_*.tif")
    #     tile_files = glob.glob(search_pattern)
        
    #     if not tile_files:
    #         raise FileNotFoundError(f"No tile files found in {self.out_dir} matching pattern 'complexity_map_*.tif'")
        
    #     print(f"Found {len(tile_files)} tile files to merge")
        
    #     # Get metadata from the first tile to determine dimensions
    #     with rasterio.open(tile_files[0]) as src:
    #         meta = src.meta.copy()
    #         num_bands = src.count  # Number of kernel sizes
        
    #     # Calculate number of tiles to process in each batch based on memory constraints
    #     # Rough estimation: float32 = 4 bytes per pixel
    #     bytes_per_pixel = 4  # float32
    #     # Estimate average tile size from first tile
    #     with rasterio.open(tile_files[0]) as sample:
    #         avg_tile_height = sample.height
    #         avg_tile_width = sample.width
        
    #     # Calculate memory usage per tile (in bytes)
    #     tile_memory_usage = avg_tile_height * avg_tile_width * num_bands * bytes_per_pixel
        
    #     # Calculate max tiles per batch
    #     max_memory_bytes = max_memory_usage_gb * 1024 * 1024 * 1024
    #     max_tiles_per_batch = max(1, int(max_memory_bytes / tile_memory_usage))
        
    #     print(f"Processing tiles in batches of {max_tiles_per_batch} to fit within {max_memory_usage_gb}GB memory limit")
        
    #     # Process tiles in batches
    #     mosaic_datasets = []
    #     for band in range(1, num_bands + 1):
    #         print(f"Processing band {band} of {num_bands} (kernel size: {self.kernel_sizes[band-1]})")
            
    #         # Process tiles in batches to manage memory
    #         num_batches = math.ceil(len(tile_files) / max_tiles_per_batch)
    #         merged_data = None
    #         merged_transform = None
            
    #         for batch_idx in range(num_batches):
    #             start_idx = batch_idx * max_tiles_per_batch
    #             end_idx = min((batch_idx + 1) * max_tiles_per_batch, len(tile_files))
    #             batch_files = tile_files[start_idx:end_idx]
                
    #             print(f"  Processing batch {batch_idx+1}/{num_batches} ({len(batch_files)} tiles)")
                
    #             # Open all files in the batch for this band
    #             batch_sources = []
    #             for file_path in tqdm(batch_files, desc="Opening tiles"):
    #                 try:
    #                     src = rasterio.open(file_path)
    #                     # Read only the current band
    #                     band_data = src.read(band)
    #                     batch_sources.append((band_data, src.transform))
    #                     src.close()
    #                 except Exception as e:
    #                     print(f"Error opening {file_path}: {e}")
                
    #             # Merge the batch
    #             if batch_sources:
    #                 batch_merged, batch_transform = self.merge_arrays(batch_sources)
                    
    #                 # If this is the first batch, initialize merged data
    #                 if merged_data is None:
    #                     merged_data = batch_merged
    #                     merged_transform = batch_transform
    #                 else:
    #                     # Merge this batch with previous batches
    #                     # This requires handling spatial alignment
    #                     combined_sources = [(merged_data, merged_transform), (batch_merged, batch_transform)]
    #                     merged_data, merged_transform = self.merge_arrays(combined_sources)
                
    #             # Clear batch data to free memory
    #             batch_sources = None
            
    #         # Store the merged band
    #         mosaic_datasets.append((merged_data, merged_transform))
        
    #     # Determine final dimensions and transform from the merged results
    #     merged_height, merged_width = mosaic_datasets[0][0].shape
    #     merged_transform = mosaic_datasets[0][1]
        
    #     # Update metadata for the output file
    #     meta.update({
    #         'driver': 'GTiff',
    #         'height': merged_height,
    #         'width': merged_width,
    #         'count': num_bands,
    #         'transform': merged_transform
    #     })
        
    #     # Create the output file and write each band
    #     with rasterio.open(output_path, 'w', **meta) as dst:
    #         for band_idx, (band_data, _) in enumerate(mosaic_datasets):
    #             dst.write(band_data, band_idx + 1)
                
    #         # Add band descriptions
    #         for band_idx, kernel_size in enumerate(self.kernel_sizes):
    #             dst.set_band_description(band_idx + 1, f"Complexity (kernel={kernel_size})")
        
    #     print(f"Mosaic successfully created: {output_path}")
    #     print(f"Dimensions: {merged_width}x{merged_height}, {num_bands} bands")
        
    #     return output_path

    # def merge_arrays(self, datasets):
    #     """
    #     Merge multiple arrays with their transforms into a single array.
        
    #     Parameters:
    #     -----------
    #     datasets : list of tuples
    #         List of (data, transform) tuples
        
    #     Returns:
    #     --------
    #     tuple : (merged_data, merged_transform)
    #     """
    #     import numpy as np
    #     from rasterio.merge import merge
        
    #     # Create a list to store the MemoryFile objects so they stay in scope
    #     memfiles = []
    #     rasterio_datasets = []
        
    #     # Convert datasets format to what rasterio.merge expects
    #     for idx, (data, transform) in enumerate(datasets):
    #         # Create a MemoryFile-based dataset with the array data
    #         from rasterio.io import MemoryFile
    #         memfile = MemoryFile()
    #         memfiles.append(memfile)  # Keep reference to prevent garbage collection
            
    #         with memfile.open(
    #             driver='GTiff',
    #             height=data.shape[0],
    #             width=data.shape[1],
    #             count=1,
    #             dtype=data.dtype,
    #             transform=transform,
    #             crs=None
    #         ) as dataset:
    #             dataset.write(data, 1)
            
    #         # Get a copy of the dataset from the MemoryFile
    #         rasterio_datasets.append(memfile.open())
        
    #     try:
    #         # Merge the datasets
    #         merged_data, merged_transform = merge(rasterio_datasets)
    #         return merged_data[0], merged_transform
    #     finally:
    #         # Close all resources
    #         for dataset in rasterio_datasets:
    #             dataset.close()
    #         for memfile in memfiles:
    #             memfile.close()