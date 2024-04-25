#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkIdentityTransform.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"

#include <iostream>

using namespace std;
using namespace itk;

int main(int argc, char * argv[])
{

 typedef float voxelType; 

 typedef Image<voxelType, 3> imageType;
 typedef ImageFileReader<imageType> imageReaderType; 
 
 typedef IdentityTransform<double, 3> transformType;
 
 typedef NearestNeighborInterpolateImageFunction<imageType, double> interpolatorType;
 
 typedef ResampleImageFilter<imageType, imageType> resamplerType;

 // Load the input image
 imageReaderType::Pointer reader = imageReaderType::New();
 reader->SetFileName(argv[4]);
 reader->Update();
 
 // Instantiate the transformation
 transformType::Pointer transform = transformType::New();
 transform->SetIdentity();
 
 // Instantiate the B-Spline interpolator
 interpolatorType::Pointer interpolator = interpolatorType::New();

 //Get new dimensions
int newX = atoi(argv[1]);
int newY = atoi(argv[2]);
int newZ = atoi(argv[3]);
 
 // Instantiate the resampler
 Size<3> size = {newX, newY, newZ};
 resamplerType::Pointer resampler = resamplerType::New();
 resampler->SetTransform(transform);
 resampler->SetInterpolator(interpolator);
 resampler->SetOutputOrigin(reader->GetOutput()->GetOrigin());
 resampler->SetSize(size);
 double spacing[3];
 spacing[0] = reader->GetOutput()->GetSpacing()[0] * (double)reader->GetOutput()->GetLargestPossibleRegion().GetSize()[0]/(double)size[0] ; 
 spacing[1] = reader->GetOutput()->GetSpacing()[1] * (double)reader->GetOutput()->GetLargestPossibleRegion().GetSize()[1]/(double)size[1] ; 
 spacing[2] = reader->GetOutput()->GetSpacing()[2] * (double)reader->GetOutput()->GetLargestPossibleRegion().GetSize()[2]/(double)size[2] ; 
 resampler->SetOutputSpacing(spacing);
 resampler->SetInput(reader->GetOutput());
 resampler->SetOutputDirection(reader->GetOutput()->GetDirection());
 resampler->Update();


 // Write the resampled image
 typedef ImageFileWriter<imageType> writerType;
 writerType::Pointer writer = writerType::New();
 writer->SetFileName(argv[5]);
 writer->SetInput(resampler->GetOutput());
 writer->Update();
 
 return 0;

}