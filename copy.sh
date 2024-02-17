for file in "checkpoints_flan_t5_large_all"/*; do
    export bs=$(basename "$file")
    echo $bs
    
    
    mkdir "checkpoints_all_filtered"/$bs
    cp $file/* "checkpoints_all_filtered"/$bs
    
done