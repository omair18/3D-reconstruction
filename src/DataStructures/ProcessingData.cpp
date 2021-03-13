#include "ProcessingData.h"

namespace DataStructures
{

ProcessingData::ProcessingData(const DataStructures::ProcessingData &other)
{

}

ProcessingData::ProcessingData(ProcessingData &&other) noexcept
{

}

ProcessingData &ProcessingData::operator=(const ProcessingData& other)
{
    return *this;
}

ProcessingData &ProcessingData::operator=(ProcessingData &&other) noexcept
{
    return *this;
}


}