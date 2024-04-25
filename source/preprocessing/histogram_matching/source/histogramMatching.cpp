
#include <iostream>
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"
#include "itkHistogramMatchingImageFilter.h"

using namespace std;
using namespace itk;


int main(int argc, char* argv[])
{

typedef float pixelType;
typedef Image<pixelType, 3> imageType;

typedef ImageFileReader<imageType> readerType;
typedef ImageFileWriter<imageType> writerType;

readerType::Pointer reader1 = readerType::New();
readerType::Pointer  reader2 = readerType::New();
writerType::Pointer writer = writerType::New();

reader1->SetFileName(argv[1]); //Input image
reader1->Update();

reader2->SetFileName(argv[2]); //Reference image
reader2->Update();

typedef HistogramMatchingImageFilter<imageType, imageType> histogramMatchingType;
histogramMatchingType::Pointer histogram = histogramMatchingType::New();

histogram->SetReferenceImage(reader2->GetOutput());
histogram->SetInput(reader1->GetOutput());
histogram->SetNumberOfHistogramLevels(1024);
histogram->SetNumberOfMatchPoints(15);
histogram->ThresholdAtMeanIntensityOn();
histogram->Update();

writer->SetFileName(argv[3]);
writer->SetInput(histogram->GetOutput());
writer->Update();

return 1;
}