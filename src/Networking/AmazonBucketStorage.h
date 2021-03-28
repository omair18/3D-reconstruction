#ifndef AMAZON_BUCKET_STORAGE_H
#define AMAZON_BUCKET_STORAGE_H

namespace Networking
{

class AmazonBucketStorage
{

public:

    AmazonBucketStorage();

    ~AmazonBucketStorage();

private:

    void InitializeAwsSdk();

    inline static bool isAwsSdkInitialized = false;

};

}

#endif // AMAZON_BUCKET_STORAGE_H
