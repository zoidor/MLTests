#include "tiff_utils.h"

#include <tiffio.h>

class TiffBuffer
{
	public:

		enum TiffChannel : std::size_t{
			Red = 0,
			Green = 1,
			Blue = 2,
			Alpha = 3
		};

		TiffBuffer(){}
		TiffBuffer(std::size_t w, std::size_t h, std::size_t channels) 
				: pData{(std::uint32_t*)_TIFFmalloc(w * h * sizeof(std::uint32_t))},
				  w{w}, h{h}, channels{channels}
		{} 
		
		~TiffBuffer(){
			if(pData)
				_TIFFfree(pData);
			pData = nullptr;
		}

		
		TiffBuffer(const TiffBuffer& other) : 
				pData{(std::uint32_t*)_TIFFmalloc(other.w * other.h * sizeof(std::uint32_t))},
				w{other.w}, h{other.h} 
		{
			_TIFFmemcpy(pData, other.pData, sizeof(std::uint32_t) * w * h);
		}

		TiffBuffer& operator=(const TiffBuffer& other){
			if(this == &other) return *this;

			this->~TiffBuffer();
			pData = (std::uint32_t*)_TIFFmalloc(other.w * other.h * sizeof(std::uint32_t));
			w = other.w;
			h = other.h;
			_TIFFmemcpy(pData, other.pData, sizeof(std::uint32_t) * w * h);
			return *this;
		}
		
		TiffBuffer(TiffBuffer&& other){
			std::swap(pData, other.pData);	
			std::swap(w, other.w);
			std::swap(h, other.h);			
		}

		TiffBuffer& operator=(TiffBuffer&& other){		
			if(this == &other) return *this;
			
			this->~TiffBuffer();
			pData = other.pData;
			other.pData = nullptr;
			w = other.w;
			h = other.h;
			return *this;
		}

		std::uint32_t* getRaster() noexcept {return pData;}
		const std::uint32_t* getRaster() const noexcept {return pData;}

		bool isEmpty() const noexcept {return (w*h) == 0;}

		std::size_t getChannels() const noexcept {return channels;}
		std::size_t getW() const noexcept {return w;}
		std::size_t getH() const noexcept {return h;}

		std::uint8_t get(const std::size_t x, const std::size_t y, const TiffChannel channel) const noexcept
		{
			if(x >= w || y >= h) return 0;
			
			const auto val = pData[x + y * w]; 

			switch(channel){
				case Red:
					return TIFFGetR(val);
				case Green:
					return TIFFGetG(val);
				case Blue:
					return TIFFGetB(val);
				case Alpha:
					return TIFFGetA(val);
				default:
					return 0;
			}
		}

	private:
 		std::uint32_t * pData{nullptr};
		std::size_t w{0};
		std::size_t h{0};
		std::size_t channels{0};
};

class TiffReadFileHandle{

	public:
		TiffReadFileHandle(const char * path) : pfile{TIFFOpen(path, "r")} {}

		~TiffReadFileHandle(){
			if(pfile)
				TIFFClose(pfile);
			pfile = nullptr;
		}

		//delete copy
		TiffReadFileHandle(const TiffReadFileHandle& other) = delete;
		TiffReadFileHandle& operator=(const TiffReadFileHandle& other) = delete;
		
		//move 
		TiffReadFileHandle(TiffReadFileHandle&& other){
			std::swap(pfile, other.pfile);		
		}
		TiffReadFileHandle& operator=(TiffReadFileHandle&& other){		
			if(this == &other) return *this;
			
			this->~TiffReadFileHandle();
			pfile = other.pfile;
			other.pfile = nullptr;
			return *this;
		}

		TiffBuffer read(){
			if(!pfile) return TiffBuffer{};
			
			std::uint32_t w{0}, h{0};
			std::uint16_t channels{0};
			TIFFGetField(pfile, TIFFTAG_IMAGEWIDTH, &w);
			TIFFGetField(pfile, TIFFTAG_IMAGELENGTH, &h);
			TIFFGetField(pfile, TIFFTAG_SAMPLESPERPIXEL, &channels);			

			if((w * h) == 0) return TiffBuffer{};
				
			TiffBuffer buff{w, h, channels}; 
			if (!TIFFReadRGBAImage(pfile, w, h, buff.getRaster(), 0)) return TiffBuffer{};
			return buff;
		}

		bool ok() const {return pfile != nullptr;}


	private:
		TIFF * pfile{nullptr};

};


tensorflow::Tensor segmentation::readTiffImage(const char * filePath, 
                     const std::size_t xMin, const std::size_t yMin, 
                     const std::int64_t cropped_h, const std::int64_t cropped_w){

	if(cropped_w <= 0) return {};
	if(cropped_h <= 0) return {};	

	TiffReadFileHandle file(filePath);
	
	if(!file.ok()) return {};

	auto buff = file.read();
	
	if(buff.isEmpty()) return {};

	const int batch_index = 1;
        tensorflow::TensorShape shape{batch_index, cropped_h, cropped_w, static_cast<std::int64_t>(buff.getChannels())};
        tensorflow::Tensor out_tensor{tensorflow::DT_FLOAT, shape};
	
	const auto xMax = xMin + cropped_w;
	const auto yMax = yMin + cropped_h;

	auto pData = out_tensor.flat<float>().data();

	using typeUndEnum = typename std::underlying_type<TiffBuffer::TiffChannel>::type;
	std::size_t i = 0;
	for(std::size_t x = xMin; x < xMax; ++x){
		for(std::size_t y = yMin; y < yMax; ++y){
			for(typeUndEnum channel = 0; channel < buff.getChannels(); ++channel){
				pData[++i] = buff.get(x, y,  static_cast<TiffBuffer::TiffChannel>(channel)); 
			} 							
		}
	} 

	return out_tensor;
}
