#!/bin/bash

root_dir="../data/raw/MoNuSAC/MoNuSAC Testing Data and Annotations"

output_dir="../data/processed/MoNuSAC_test/MoNuSAC Testing Data and Annotations png"

mkdir -p "$output_dir"

find "$root_dir" -type f -name '*.tif' | while read tif_file; do
  base_name=$(basename "$tif_file" .tif)
  
  png_file="$output_dir/$base_name.png"
  
  convert "$tif_file" "$png_file"
  
  echo "Converted $tif_file to $png_file"
done