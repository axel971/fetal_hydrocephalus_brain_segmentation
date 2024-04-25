#include <iostream>
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkDiscreteGaussianImageFilter.h"

using namespace std;
using namespace itk;

int main(int argc, char* argv[])
{
 using pixelType = float;
 
using imageType = Image<pixelType, 3>;

// Intentiate reader and writer types for the input and output images
using readerType = ImageFileReader<imageType>;
using writerType = ImageFileWriter<imageType>;

// Intentiate iterator type
using iteratorType = ImageRegionIterator<imageType>;

// Read the input image
readerType::Pointer reader = readerType::New();
reader->SetFileName(argv[1]);
reader->Update();

// Create the output image
imageType::Pointer outputImg = imageType::New();
outputImg->SetRegions(reader->GetOutput()->GetLargestPossibleRegion());
outputImg->SetSpacing(reader->GetOutput()->GetSpacing());
outputImg->SetOrigin(reader->GetOutput()->GetOrigin());
outputImg->SetDirection(reader->GetOutput()->GetDirection());
outputImg->Allocate();

// Assign an iterator to the input images and output images
iteratorType iterInputImg(reader->GetOutput(), reader->GetOutput()->GetLargestPossibleRegion());
iteratorType iterOutputImg(outputImg, outputImg->GetLargestPossibleRegion());

// Modify the label of the delineations
for(iterInputImg.GoToBegin(), iterOutputImg.GoToBegin(); !iterInputImg.IsAtEnd(), !iterOutputImg.IsAtEnd(); ++iterInputImg, ++iterOutputImg)
{
		if(iterInputImg.Get() == 5 ) //Lateral Ventricles
			iterOutputImg.Set(1);
					
		else
			iterOutputImg.Set(0); //background
}

// Write on the hard disk the output image
writerType::Pointer writer = writerType::New();
writer->SetFileName(argv[2]);
writer->SetInput(outputImg);
writer->Update();

return 1;
}