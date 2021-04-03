#include "AmazonBucketAPI.h"

namespace Networking
{

AmazonBucketAPI::AmazonBucketAPI()
{

}

AmazonBucketAPI::~AmazonBucketAPI()
{
    if(isAwsSdkInitialized)
    {
        isAwsSdkInitialized = false;

    }
}

void AmazonBucketAPI::InitializeAwsSdk()
{

}


}
