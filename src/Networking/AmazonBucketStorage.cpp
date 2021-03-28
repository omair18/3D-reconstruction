#include "AmazonBucketStorage.h"

namespace Networking
{

AmazonBucketStorage::AmazonBucketStorage()
{

}

AmazonBucketStorage::~AmazonBucketStorage()
{
    if(isAwsSdkInitialized)
    {
        isAwsSdkInitialized = false;

    }
}


}
