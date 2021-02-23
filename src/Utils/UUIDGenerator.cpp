#include <boost/uuid/uuid.hpp>
#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/lexical_cast.hpp>

#include "UUIDGenerator.h"

std::string UUIDGenerator::GenerateUUID()
{
    return boost::lexical_cast<std::string>(boost::uuids::uuid(boost::uuids::random_generator()()));
}
