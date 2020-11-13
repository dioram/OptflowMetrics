#include <readers/Readers.h>
#include "SintelReader.h"
#include "KittiReader.h"

IReaderPtr Readers::makeKittiReader(const std::string& dir) {
	return std::make_shared<KittyReader>(dir);
}

IReaderPtr Readers::makeSintelReader(const std::string& dir, const RenderingType& type) {
	return std::make_shared<SintelReader>(dir, type);
}