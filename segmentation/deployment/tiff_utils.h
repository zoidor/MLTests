#ifndef SEGMENTATION_TIFF_UTILS_H
#define SEGMENTATION_TIFF_UTILS_H

#include "tensorflow/core/framework/tensor.h"

namespace Segmentation{
	tensorflow::Tensor readTiffImage(const char * filePath, 
                     const std::size_t xMin, const std::size_t yMin, 
                     const std::int64_t cropped_h, const std::int64_t cropped_w);
}
#endif //SEGMENTATION_TIFF_UTILS_H
