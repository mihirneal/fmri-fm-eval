#!/bin/bash

sub_list_path="metadata/adhd200_subject_list.txt"
rm $sub_list_path 2>/dev/null

while read subdir; do
    # data/fmriprep/WashU/sub-0015060
    dataset=$(echo $subdir | cut -d / -f 3)
    sub=$(echo $subdir | cut -d / -f 4)
    sub=${sub#sub-}
    echo $dataset $sub >> $sub_list_path
done < <(find data/RawDataBIDS -type d -name 'sub-*' | sort)
