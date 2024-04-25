

listID='/home/axel/dev/fetal_hydrocephalus_segmentation/data/list_fetus_IDs.csv'

inputMRIFolder='/home/axel/dev/fetal_hydrocephalus_segmentation/data/corrected_data/MRI'
outputMRIFolder='/home/axel/dev/fetal_hydrocephalus_segmentation/data/preprocessing/MRI/data'

inputDelineationFolder='/home/axel/dev/fetal_hydrocephalus_segmentation/data/corrected_data/Segmentation'
outputDelineationFolder='/home/axel/dev/fetal_hydrocephalus_segmentation/data/preprocessing/delineationn/data'


for ID in `cat $listID`
do
	ID=`echo $ID | cut -d ',' -f2`
	
	if [[ $ID != "ID" ]]
	then
		echo $ID
		cp  ${inputMRIFolder}/${ID}*.nii.gz ${outputMRIFolder}/${ID}.nii.gz
		cp  ${inputDelineationFolder}/${ID}*.nii.gz ${outputDelineationFolder}/${ID}_delineation.nii.gz
	fi
done