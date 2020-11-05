#include <readers/Readers.h>
#include "CintelReader.h"
#include "KittiReader.h"

IReaderPtr Readers::makeKittiReader(const std::string& dir) {
	return std::make_shared<KittyReader>(dir);
}

IReaderPtr Readers::makeCintelReader(const std::string& dir) {
	throw std::runtime_error("not implemented");
}