
listID='/home/axel/dev/fetal_hydrocephalus_segmentation/data/list_fetus_IDs.csv'

inputMRIFolder='/home/axel/dev/fetal_hydrocephalus_segmentation/data/preprocessing/MRI/data'
histogramMatchedMRIFolder='/home/axel/dev/fetal_hydrocephalus_segmentation/data/preprocessing/MRI/histogram_matching'

inputDelineationFolder='/home/axel/dev/fetal_hydrocephalus_segmentation/data/preprocessing/delineationn/data'
relabeledDelineationFolder='/home/axel/dev/fetal_hydrocephalus_segmentation/data/preprocessing/delineationn/relabeling'


for ID in `cat $listID`
do
	ID=`echo $ID | cut -d ',' -f2`
	
	if [[ $ID != "ID" ]]
	then
		echo "Delineation relabeling for subject: " $ID
		./relabeling/build/relabeling ${inputDelineationFolder}/${ID}_delineation.nii.gz ${relabeledDelineationFolder}/${ID}_relabeled_delineation.nii
		
	fi
done


# Histogram matching
tmp=`cat $listID`
tmp=(${tmp})	
reference_subject=`echo ${tmp[1]} | cut -d ',' -f2`

echo "Reference subject for histogram matching is: " ${reference_subject}

for ID in `cat $listID`
do
	ID=`echo $ID | cut -d ',' -f2`
	
	if [[ $ID != "ID" ]]
	then
		echo "Histogram matching for subject: " $ID
		./histogram_matching/build/histogramMatching ${inputMRIFolder}/${ID}.nii.gz ${inputMRIFolder}/${reference_subject}.nii.gz ${histogramMatchedMRIFolder}/${ID}_preproc.nii
		
	fi
done