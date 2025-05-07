import os
import glob
import argparse
from osgeo import gdal

def create_virtual_mosaic(input_dir, output_file, file_extension='.tif'):
    """
    Create a virtual mosaic (VRT) file from all raster files with the specified extension in a directory.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing the input raster files
    output_file : str
        Path to the output VRT file
    file_extension : str
        File extension to filter for (e.g., '.tif', '.img')
    """
    # Make sure the input directory exists
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Find all files with the specified extension
    search_pattern = os.path.join(input_dir, f"*{file_extension}")
    input_files = glob.glob(search_pattern)
    
    if not input_files:
        raise ValueError(f"No files with extension {file_extension} found in {input_dir}")
    
    print(f"Found {len(input_files)} files with extension {file_extension}")
    
    # Create the VRT options object correctly
    vrt_options = gdal.BuildVRTOptions(resampleAlg='nearest')
    
    # Build the VRT mosaic
    print(f"Creating virtual mosaic at: {output_file}")
    vrt = gdal.BuildVRT(output_file, input_files, options=vrt_options)
    
    # This is important to flush the data to disk
    vrt = None
    
    print(f"Virtual mosaic created successfully: {output_file}")
    return output_file

def get_raster_info(file_path):
    """Get and print basic information about a raster file."""
    try:
        dataset = gdal.Open(file_path)
        if dataset is None:
            print(f"Failed to open {file_path}")
            return
        
        print(f"\nRaster information for: {file_path}")
        print(f"Format: {dataset.GetDriver().ShortName}")
        print(f"Size: {dataset.RasterXSize} x {dataset.RasterYSize} x {dataset.RasterCount}")
        
        # Get projection and extent
        projection = dataset.GetProjection()
        if projection:
            print(f"Projection is DEFINED")
        else:
            print(f"Projection is NOT DEFINED")
            
        # Get geotransform
        geotransform = dataset.GetGeoTransform()
        if geotransform:
            print(f"Origin: ({geotransform[0]}, {geotransform[3]})")
            print(f"Pixel Size: ({geotransform[1]}, {geotransform[5]})")
        else:
            print("Geotransform is NOT DEFINED")
            
        # Get band information
        for i in range(1, dataset.RasterCount + 1):
            band = dataset.GetRasterBand(i)
            print(f"Band {i}: Type={gdal.GetDataTypeName(band.DataType)}")
            
            try:
                minimum, maximum, mean, stddev = band.GetStatistics(True, True)
                print(f"  Min={minimum:.3f}, Max={maximum:.3f}, Mean={mean:.3f}, StdDev={stddev:.3f}")
            except:
                print("  Statistics not available")
            
            if band.GetNoDataValue() is not None:
                print(f"  NoData Value: {band.GetNoDataValue()}")
            
        # Close the dataset
        dataset = None
        
    except Exception as e:
        print(f"Error getting information for {file_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a virtual mosaic from raster files.")
    parser.add_argument("input_dir", help="Directory containing the input raster files")
    parser.add_argument("output_file", help="Output VRT file path")
    parser.add_argument("--extension", default=".tif", help="File extension to filter for (default: .tif)")
    parser.add_argument("--show-info", action="store_true", help="Show information about the created mosaic")
    
    args = parser.parse_args()
    
    # Create the virtual mosaic
    vrt_file = create_virtual_mosaic(args.input_dir, args.output_file, args.extension)
    
    # Optionally show information about the created mosaic
    if args.show_info:
        get_raster_info(vrt_file)