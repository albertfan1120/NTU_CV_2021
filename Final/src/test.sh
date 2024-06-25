
read -p "Please input the input file path: " data_root   
read -p "Please input the name of the sequence: " seq   
read -p "Please input the number of clips in the sequence: " act_num
echo $'\n'"filepath = "$data_root"/"$seq$'\n'
echo 'total '$act_num 'clips'$'\n'

# generate mask
# generate confidence
python3 test.py --data_root ${data_root} --seq ${seq} --act_num ${act_num}

# mask post-processing
python3 postprocessing.py --seq ${seq}
# confidence post-processing
python3 postprocessing_label.py --seq ${seq}


