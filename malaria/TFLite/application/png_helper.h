#ifndef MALARIA_PNG_HELPER_H
#define MALARIA_PNG_HELPER_H

#include <vector>
#include <cinttypes>
#include <string>

namespace malaria{

std::vector<float> read_png_and_resize(const std::string& path, const int wanted_height, const int wanted_width, const int wanted_channels);

}
#endif //MALARIA_PNG_HELPER_H

