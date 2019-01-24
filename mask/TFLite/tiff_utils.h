#ifndef SEGMENTATION_TIFF_UTILS_H
#define SEGMENTATION_TIFF_UTILS_H

#include <vector>
#include <cinttypes>

namespace segmentation{
        std::vector<std::uint8_t> readTiffImage(const char * filePath, 
                     const std::size_t xMin, const std::size_t yMin, 
                     const std::int64_t cropped_h, const std::int64_t cropped_w);
}
#endif //SEGMENTATION_TIFF_UTILS_H
